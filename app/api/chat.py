# app/api/chat.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List
from app.rag.pipeline import run_rag_pipeline

router = APIRouter()

# Đúng schema API của Section 26.2
class ChatRequest(BaseModel):
    session_id: str
    question: str
    code: Optional[str] = None
    log: Optional[str] = None

@router.post("/chat")
async def api_chat(request: ChatRequest):
    # Trả đúng Response Schema theo yêu cầu
    answer, sources = run_rag_pipeline(
        session_id=request.session_id,
        question=request.question,
        code=request.code,
        log=request.log
    )
    
    return {
        "answer": answer,
        "sources": sources,
        "session_id": request.session_id
    }