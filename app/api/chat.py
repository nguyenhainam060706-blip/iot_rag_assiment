from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from app.rag.pipeline import run_rag_pipeline 

router = APIRouter()

# 1. Định nghĩa Input (Request Schema)
class ChatRequest(BaseModel):
    session_id: str
    question: str = Field(..., min_length=1, description="Câu hỏi không được để trống")
    code: Optional[str] = None
    log: Optional[str] = None

# 2. Định nghĩa Source Citation (Theo mục 27 của Spec)
class SourceItem(BaseModel):
    document: str
    page: int
    score: float

# 3. Định nghĩa Output (Response Schema)
class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    session_id: str

@router.post("", response_model=ChatResponse)
async def api_chat(request: ChatRequest):
    try:
        # GỌI HÀM THỰC TẾ: Chuyền dữ liệu từ API xuống pipeline
        # (Lưu ý: Nếu hàm run_rag_pipeline của bạn dùng 'async def', 
        # hãy thêm chữ 'await' ở đầu: await run_rag_pipeline(...))
        answer, raw_sources = run_rag_pipeline(
            session_id=request.session_id,
            question=request.question,
            code=request.code,
            log=request.log
        )
        
        # CHUẨN HÓA DỮ LIỆU NGUỒN (SOURCES):
        # Đảm bảo dữ liệu từ pipeline trả về (thường là list of dicts) 
        # được ép kiểu chuẩn xác sang Pydantic model 'SourceItem'
        formatted_sources = [
            SourceItem(**src) if isinstance(src, dict) else src 
            for src in raw_sources
        ]
        
        return ChatResponse(
            answer=answer,
            sources=formatted_sources,
            session_id=request.session_id
        )
        
    except Exception as e:
        # Xử lý lỗi theo Spec mục 30
        raise HTTPException(
            status_code=500, 
            detail={"error": "LLM_SERVICE_UNAVAILABLE", "message": str(e)}
        )