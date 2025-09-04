import os
from datetime import datetime
import logging
from dotenv import load_dotenv
import tempfile
import shutil
from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # Updated import
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.documents import Document


from .customTypes import (
    Utterance
)
# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, persist_directory: str = "./database"):
        self.persist_directory = persist_directory
        self._ensure_database_directory()
        self.embedding = OpenAIEmbeddings(model='text-embedding-3-large')
        self.retriever = self.initialize_hybrid_retriever()
    
    def _ensure_database_directory(self):
        """Check if database directory exists, create it if it doesn't"""
        if not os.path.exists(self.persist_directory):
            try:
                os.makedirs(self.persist_directory, exist_ok=True)
                print(f"Created database directory: {self.persist_directory}")
            except Exception as e:
                print(f"Error creating database directory {self.persist_directory}: {e}")
                raise e
        else:
            print(f"Database directory already exists: {self.persist_directory}")
        
    def initialize_hybrid_retriever(self):
        """Initialize hybrid retriever combining semantic and keyword search"""
        
        # Load Chroma vector database (for semantic search)
        vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)
        semantic_retriever = vectordb.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 5}  # Get more results for better fusion
        )
        
        # Get documents from Chroma for BM25 initialization
        try:
            # Retrieve documents to initialize BM25
            all_docs = vectordb.get()
            if all_docs and 'documents' in all_docs and all_docs['documents']:
                # Create Document objects for BM25
                from langchain_core.documents import Document
                documents = [Document(page_content=doc) for doc in all_docs['documents']]
                
                # Initialize BM25 retriever (for keyword search)
                bm25_retriever = BM25Retriever.from_documents(documents)
                bm25_retriever.k = 5  # Number of documents to retrieve
                
                # Create hybrid retriever using EnsembleRetriever
                hybrid_retriever = EnsembleRetriever(
                    retrievers=[semantic_retriever, bm25_retriever],
                    weights=[0.6, 0.4],  # 60% semantic, 40% keyword search
                )
                
                print("Hybrid retriever initialized successfully")
                return hybrid_retriever
            else:
                print("No documents found in Chroma DB. Using semantic search only.")
                return semantic_retriever
                
        except Exception as e:
            print(f"Could not initialize hybrid search: {e}. Using semantic search only.")
            return semantic_retriever


    def last_msg(self, transcript_messages):
        """Extract the last user message from transcript"""
        for message in reversed(transcript_messages):
            if message['role'] == 'user':
                return message['content']
        return ""

    def update_search_type(self, search_type: str = "hybrid"):
        """Update the search type - hybrid, semantic, or keyword"""
        
        if search_type == "semantic":
            vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)
            self.retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            print("Switched to semantic search only")
            
        elif search_type == "keyword":
            # This would require BM25 only setup
            try:
                vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)
                all_docs = vectordb.get()
                if all_docs and 'documents' in all_docs and all_docs['documents']:
                    from langchain_core.documents import Document
                    documents = [Document(page_content=doc) for doc in all_docs['documents']]
                    self.retriever = BM25Retriever.from_documents(documents)
                    self.retriever.k = 3
                    print("Switched to keyword search only")
                else:
                    print("No documents found for keyword search")
            except Exception as e:
                print(f"Could not initialize keyword search: {e}")
                
        else:  # default to hybrid
            self.retriever = self.initialize_hybrid_retriever()
            print("Using hybrid search")

    async def add_document(self, file_name: str, content: bytes, content_type: str, session_id: str) -> int:
        """Process and add a document to the RAG system"""
        try:
            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=self._get_file_extension(file_name)) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            # Load and process document based on type
            documents = self._load_document(temp_file_path, content_type)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            if not documents:
                raise Exception("No content extracted from document")
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            split_docs = text_splitter.split_documents(documents)
            
            # Add metadata to documents
            for doc in split_docs:
                doc.metadata.update({
                    'session_id': session_id,
                    'file_name': file_name,
                    'uploaded_at': datetime.now().isoformat(),
                    'content_type': content_type
                })
            
            # Add to Chroma vector database
            vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)
            vectordb.add_documents(split_docs)
            
            # Reinitialize retriever to include new documents
            self.retriever = self.initialize_hybrid_retriever()
            
            print(f"Successfully processed {len(split_docs)} document chunks from {file_name}")
            return len(split_docs)
            
        except Exception as e:
            print(f"Error processing document {file_name}: {e}")
            raise e

    def _load_document(self, file_path: str, content_type: str) -> List[Document]:
        """Load document based on content type"""
        try:
            if content_type == 'application/pdf':
                loader = PyPDFLoader(file_path)
            elif content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                loader = UnstructuredWordDocumentLoader(file_path)
            elif content_type in ['text/plain', 'text/markdown']:
                loader = TextLoader(file_path)
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            return loader.load()
            
        except Exception as e:
            print(f"Error loading document: {e}")
            raise e

    def _get_file_extension(self, file_name: str) -> str:
        """Get file extension from filename"""
        return os.path.splitext(file_name)[1]

    def get_session_documents(self, session_id: str) -> List[dict]:
        """Get all documents for a specific session"""
        try:
            vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)
            # This is a simplified approach - in production you'd want a more sophisticated query
            all_docs = vectordb.get()
            
            if not all_docs or 'metadatas' not in all_docs:
                return []
            
            session_docs = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if metadata and metadata.get('session_id') == session_id:
                    session_docs.append({
                        'file_name': metadata.get('file_name', 'Unknown'),
                        'uploaded_at': metadata.get('uploaded_at', 'Unknown'),
                        'content_type': metadata.get('content_type', 'Unknown'),
                        'chunk_index': i
                    })
            
            return session_docs
            
        except Exception as e:
            print(f"Error getting session documents: {e}")
            return []

    def delete_session_documents(self, session_id: str) -> int:
        """Delete all documents for a specific session"""
        try:
            vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)
            all_docs = vectordb.get()
            
            if not all_docs or 'metadatas' not in all_docs:
                return 0
            
            # Find indices of documents to delete
            indices_to_delete = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if metadata and metadata.get('session_id') == session_id:
                    indices_to_delete.append(i)
            
            if indices_to_delete:
                # Delete documents by index
                vectordb.delete(ids=[all_docs['ids'][i] for i in indices_to_delete])
                
                # Reinitialize retriever
                self.retriever = self.initialize_hybrid_retriever()
                
                print(f"Deleted {len(indices_to_delete)} documents from session {session_id}")
                return len(indices_to_delete)
            
            return 0
            
        except Exception as e:
            print(f"Error deleting session documents: {e}")
            return 0

    def get_all_documents(self) -> List[dict]:
        """Get all documents stored in the database with their metadata"""
        try:
            vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)
            all_docs = vectordb.get()
            
            if not all_docs or 'documents' not in all_docs:
                return []
            
            documents = []
            for i, (doc_content, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
                if metadata:
                    documents.append({
                        'id': all_docs['ids'][i] if 'ids' in all_docs else str(i),
                        'content': doc_content,
                        'file_name': metadata.get('file_name', 'Unknown'),
                        'session_id': metadata.get('session_id', 'Unknown'),
                        'uploaded_at': metadata.get('uploaded_at', 'Unknown'),
                        'content_type': metadata.get('content_type', 'Unknown'),
                        'chunk_index': i
                    })
            
            print("documents retreived", documents)
            return documents
            
        except Exception as e:
            print(f"Error getting all documents: {e}")
            return []

    def get_documents_by_names(self, file_names: List[str]) -> List[dict]:
        """Get documents by specific file names"""
        try:
            all_docs = self.get_all_documents()
            return [doc for doc in all_docs if doc['file_name'] in file_names]
            
        except Exception as e:
            print(f"Error getting documents by names: {e}")
            return []

    def query_specific_documents(self, query: str, file_names: List[str]) -> List[dict]:
        """Query only specific documents by file names"""
        try:
            if not file_names:
                # If no specific documents selected, use all documents
                return self.retriever.get_relevant_documents(query)
            
            # Get documents by names
            # print(f"Querying specific documents: {file_names}")
            target_docs = self.get_documents_by_names(file_names)
            if not target_docs:
                print(f"No documents found with names: {file_names}")
                return []
            
            print(f"Target docs: {target_docs}")
            
            # Create a temporary retriever for the selected documents
            from langchain_core.documents import Document
            documents = [Document(page_content=doc['content'], metadata=doc) for doc in target_docs]
            
            # Use semantic search on selected documents
            vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)
            semantic_retriever = vectordb.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 5}
            )
            
            # Filter the retriever to only search in selected documents
            # This is a simplified approach - in production you'd want more sophisticated filtering
            relevant_docs = semantic_retriever.get_relevant_documents(query)
            
            # Filter results to only include documents from selected files
            filtered_docs = []
            for doc in relevant_docs:
                if hasattr(doc, 'metadata') and doc.metadata:
                    doc_file_name = doc.metadata.get('file_name', '')
                    if doc_file_name in file_names:
                        filtered_docs.append(doc)
            
            print(f"Retrieved {len(filtered_docs)} relevant chunks from selected documents")
            return filtered_docs
            
        except Exception as e:
            print(f"Error querying specific documents: {e}")
            return []
