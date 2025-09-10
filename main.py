from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from datetime import datetime
from dotenv import load_dotenv
import uuid

from src.customTypes import RAGRequest, RAGResponse, HealthCheckResponse, DocumentUploadResponse
from src.rag import rag
from src.ragUtils import RAGSystem

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Healthcare API",
    description="FastAPI middleware for RAG-based healthcare appointment system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rag-ai-chatbot-frontend.vercel.app/",
        # "http://localhost:3001",
        # "http://localhost:5173",
        # "http://localhost:3000"
        ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=[
        'Content-Type',
        'Authorization',
        'ngrok-skip-browser-warning',
        'Accept',
        'Origin',
        'X-Requested-With',
        'X-License-Key'
    ],
)
# Simple license key (non-expiring) loaded from env or default
LICENSE_KEY = os.getenv("LICENSE_KEY", "demo-license-123")

# Middleware to enforce license on all /api routes
@app.middleware("http")
async def license_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)  # allow CORS preflight
    if request.url.path.startswith("/api/"):
        provided = request.headers.get("x-license-key")
        if provided != LICENSE_KEY:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing license key"})
    return await call_next(request)

@app.middleware("https")
async def license_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)  # allow CORS preflight
    if request.url.path.startswith("/api/"):
        provided = request.headers.get("x-license-key")
        if provided != LICENSE_KEY:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing license key"})
    return await call_next(request)


# Global RAG system instance
rag_system = None

def get_rag_system():
    """Dependency to get RAG system instance"""
    global rag_system
    if rag_system is None:
        try:
            rag_system = RAGSystem(persist_directory="./database")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize RAG system: {str(e)}")
    return rag_system

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global rag_system
    try:
        rag_system = RAGSystem(persist_directory="./database")
        print("RAG system initialized successfully")
    except Exception as e:
        print(f"Warning: RAG system initialization failed: {e}")

@app.get("/", response_model=HealthCheckResponse)
async def root():
    """Root endpoint with health check"""
    return HealthCheckResponse(
        status="healthy",
        message="RAG Healthcare API is running",
        timestamp=datetime.now().isoformat()
    )

@app.head("/")
async def root_head():
    """Root endpoint HEAD handler for health checks"""
    return None

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check if RAG system is accessible
        rag_sys = get_rag_system()
        return HealthCheckResponse(
            status="healthy",
            message="All systems operational",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        return HealthCheckResponse(
            status="unhealthy",
            message=f"System error: {str(e)}",
            timestamp=datetime.now().isoformat()
        )

@app.head("/health")
async def health_check_head():
    """Health check HEAD handler"""
    try:
        # Quick check if RAG system is accessible
        get_rag_system()
        return None  # HEAD requests should return no body
    except Exception:
        # Let the exception propagate to return 500 status
        raise

@app.get("/api/documents/all")
async def get_all_documents(rag_sys: RAGSystem = Depends(get_rag_system)):
    """Get all documents stored in the database"""
    try:
        documents = rag_sys.get_all_documents()
        
        # Group documents by session for better organization
        sessions = {}
        for doc in documents:
            session_id = doc['session_id']
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append({
                'file_name': doc['file_name'],
                'uploaded_at': doc['uploaded_at'],
                'content_type': doc['content_type'],
                'chunk_count': 1  # Each document represents one chunk
            })
        
        # Aggregate chunk counts per file
        file_summary = {}
        for session_docs in sessions.values():
            for doc in session_docs:
                file_name = doc['file_name']
                if file_name not in file_summary:
                    file_summary[file_name] = {
                        'file_name': file_name,
                        'uploaded_at': doc['uploaded_at'],
                        'content_type': doc['content_type'],
                        'total_chunks': 0,
                        'sessions': set()
                    }
                file_summary[file_name]['total_chunks'] += doc['chunk_count']
                file_summary[file_name]['sessions'].add(doc.get('session_id', 'Unknown'))
        
        # Convert sets to lists for JSON serialization
        for file_info in file_summary.values():
            file_info['sessions'] = list(file_info['sessions'])
        
        return {
            "total_documents": len(documents),
            "unique_files": len(file_summary),
            "files": list(file_summary.values()),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)}")

@app.post("/api/rag/query", response_model=RAGResponse)
async def process_rag_query(request: RAGRequest):
    """
    Process RAG query with transcript and return AI response
    
    This endpoint processes queries using the RAG system.
    It takes a transcript of conversation and returns an AI-generated response
    based on the knowledge base.
    """
    try:
        # Validate request
        if not request.transcript:
            raise HTTPException(status_code=400, detail="Transcript cannot be empty")
        
        # Get the RAG system instance
        rag_sys = get_rag_system()
        
        # Process the RAG query with document selection
        response = await rag(request, rag_sys)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG processing failed: {str(e)}")

@app.post("/api/rag/search-type", response_model=dict)
async def update_search_type(search_type: str, rag_sys: RAGSystem = Depends(get_rag_system)):
    """
    Update the search type for RAG queries
    
    Available options:
    - "hybrid": Combines semantic and keyword search (default)
    - "semantic": Uses only vector similarity search
    - "keyword": Uses only BM25 keyword search
    """
    try:
        if search_type not in ["hybrid", "semantic", "keyword"]:
            raise HTTPException(
                status_code=400, 
                detail="Search type must be one of: hybrid, semantic, keyword"
            )
        
        rag_sys.update_search_type(search_type)
        return {
            "message": f"Search type updated to {search_type}",
            "search_type": search_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update search type: {str(e)}")

@app.get("/api/rag/status", response_model=dict)
async def get_rag_status(rag_sys: RAGSystem = Depends(get_rag_system)):
    """Get current RAG system status and configuration"""
    try:
        return {
            "status": "operational",
            "persist_directory": rag_sys.persist_directory,
            "embedding_model": "text-embedding-3-large",
            "llm_model": "gpt-4o-mini",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = None,
    rag_sys: RAGSystem = Depends(get_rag_system)
):
    """
    Upload and process a document for the RAG system
    
    This endpoint:
    1. Receives a document file
    2. Generates a session ID if not provided
    3. Processes the document and creates vector embeddings
    4. Stores the document in the knowledge base
    """
    try:
        # Validate file type
        allowed_types = [
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'text/plain',
            'text/markdown'
        ]
        
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file.content_type} not supported. Allowed types: {allowed_types}"
            )
        
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Read file content
        content = await file.read()
        
        # Process document and add to RAG system
        documents_processed = await rag_sys.add_document(
            file_name=file.filename,
            content=content,
            content_type=file.content_type,
            session_id=session_id
        )
        
        return DocumentUploadResponse(
            message=f"Document {file.filename} processed successfully",
            documents_processed=documents_processed,
            session_id=session_id,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")

@app.get("/api/documents/sessions/{session_id}")
async def get_session_documents(
    session_id: str,
    rag_sys: RAGSystem = Depends(get_rag_system)
):
    """Get all documents for a specific session"""
    try:
        documents = rag_sys.get_session_documents(session_id)
        return {
            "session_id": session_id,
            "documents": documents,
            "count": len(documents),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session documents: {str(e)}")

@app.delete("/api/documents/sessions/{session_id}")
async def delete_session_documents(
    session_id: str,
    rag_sys: RAGSystem = Depends(get_rag_system)
):
    """Delete all documents for a specific session"""
    try:
        deleted_count = rag_sys.delete_session_documents(session_id)
        return {
            "message": f"Deleted {deleted_count} documents from session {session_id}",
            "deleted_count": deleted_count,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session documents: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
