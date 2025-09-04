from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

class Utterance(BaseModel):
    role: str
    content: str

class RAGRequest(BaseModel):
    transcript: List[Utterance]
    search_type: Optional[str] = "hybrid"
    temperature: Optional[float] = 0.0
    model: Optional[str] = "gpt-4o-mini"
    selected_documents: Optional[List[str]] = []  # List of file names to focus on

class RAGResponse(BaseModel):
    response: str
    context_used: Optional[str] = None
    documents_retrieved: Optional[int] = None
    processing_time: Optional[float] = None
    search_type: Optional[str] = None
    error: Optional[str] = None

class HealthCheckResponse(BaseModel):
    status: str
    message: str
    timestamp: str

class DocumentUploadResponse(BaseModel):
    message: str
    documents_processed: int
    session_id: str
    status: str