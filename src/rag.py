import os
from .ragUtils import RAGSystem
from datetime import datetime
from .transcriptMessages import convert_transcript_to_openai_messages
from .customTypes import RAGResponse, RAGRequest

from langchain_chroma import Chroma
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

persist_directory = "./database"
ragSystem = RAGSystem(persist_directory=persist_directory)

# RAG FUNCTIONS
async def rag(request: RAGRequest, rag_sys: RAGSystem = None):
    start_time = datetime.now()
    print(f"---- Start Time: {start_time} ----")
    
    # Use provided RAG system or fallback to global
    if rag_sys is None:
        rag_sys = ragSystem
    
    try:
        print("----in rag func----")
        transcript_messages = convert_transcript_to_openai_messages(request.transcript)

        msg = rag_sys.last_msg(transcript_messages)
        print(f"{datetime.now()} - After message extraction ...")
        print(msg, "---- msg to llm----")
        
        # Check if specific documents are selected
        selected_documents = getattr(request, 'selected_documents', [])
        print(f"Selected documents: {selected_documents}")
        
        # Use hybrid retriever to get relevant documents
        try:
            if selected_documents:
                # Query only selected documents
                relevant_docs = rag_sys.query_specific_documents(msg, selected_documents)
                print(f"Retrieved {len(relevant_docs)} documents from selected files: {selected_documents}")
            else:
                # Get relevant documents using hybrid search from all documents
                relevant_docs = rag_sys.retriever.get_relevant_documents(msg)
                print(f"Retrieved {len(relevant_docs)} documents using hybrid search from all documents")
            
            # Format documents for context
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
        except Exception as e:
            print(f"Error in retrieval: {e}")
            # Fallback to direct vector store search if hybrid fails
            vectordb = Chroma(persist_directory=rag_sys.persist_directory, embedding_function=rag_sys.embedding)
            if selected_documents:
                # Fallback for selected documents
                relevant_docs = rag_sys.query_specific_documents(msg, selected_documents)
            else:
                relevant_docs = vectordb.similarity_search(msg, k=3)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Enhanced prompt template for general RAG use cases
        augmented_prompt = f"""

        # ENFORCED BEHAVIOUR:
        - You are a helpful AI assistant that provides accurate and relevant information based on the knowledge base provided.

        ###HIGHLY ENFORCED RULE:
        - Return all responses in plain text without any Markdown or special formatting symbols.
        - Generate a proper response based on the context provided.
        - Provide clear and concise answers to the user's query.
        - If the user query is not clear, ask for clarification.
        - Base your response primarily on the provided context, but you can also use your general knowledge if needed.

        <context>
        Context from knowledge base:
        {context}

        </context>
        
        <user_query>
        User Query: {msg}

        </user_query>

        <response>
        "your plain text response here"


        </response>

        """

        messages = [
            SystemMessage(content=f"You are a helpful AI assistant that provides accurate and relevant information based on the knowledge base provided."),
            HumanMessage(content=augmented_prompt)
        ]

        print(f"{datetime.now()} - After RAG ...")
        
        # Use the initialized LLM
        print(f"Model USED: {request.model}")
        chat = ChatOpenAI(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            model=request.model if hasattr(request, 'model') else 'gpt-4o-mini',
            temperature=request.temperature if hasattr(request, 'temperature') else 0,
        )
        
        print(f"{datetime.now()} - Before LLM response...")
        
        # Send to OpenAI
        res = chat(messages)
        print(f"_________LLM Response______: {res.content}")
        

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"---- End Time: {end_time} ----")
        print(f"---- Duration: {duration} ----")

        return RAGResponse(
            response=res.content,
            context_used=context,
            documents_retrieved=len(relevant_docs),
            processing_time=duration,
            search_type=request.search_type if hasattr(request, 'search_type') else "hybrid"
        )
        
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"Error in RAG processing: {e}")
        return RAGResponse(
            response="",
            error=str(e),
            processing_time=duration
        )
