from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List

from app.rag.pipeline import run_rag_pipeline
from app.config import settings
from app.core.exceptions import InvalidRequest

router = APIRouter()


# 1. Định nghĩa Input (Request Schema)
# Có giới hạn max_length theo mục 33 (Security) để tránh input quá lớn
# tràn context LLM hoặc làm chậm/DoS đơn giản.
class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    question: str = Field(
        ..., min_length=1, max_length=settings.MAX_QUESTION_LENGTH,
        description="Câu hỏi không được để trống",
    )
    code: Optional[str] = Field(None, max_length=settings.MAX_CODE_LENGTH)
    log: Optional[str] = Field(None, max_length=settings.MAX_LOG_LENGTH)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        # min_length=1 không chặn được chuỗi toàn khoảng trắng (vd: "   ")
        if not v.strip():
            raise ValueError("Câu hỏi không được để trống.")
        return v


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


# Dùng 'def' (không async) vì run_rag_pipeline là hàm đồng bộ (blocking) ->
# FastAPI tự chạy trong threadpool riêng, không block event loop chính.
@router.post("", response_model=ChatResponse)
def api_chat(request: ChatRequest):
    # KHÔNG bọc try/except Exception rồi gán cứng 1 mã lỗi (LLM_SERVICE_UNAVAILABLE)
    # cho MỌI loại lỗi -> sai bản chất, gây khó debug và sai thông tin cho frontend.
    #
    # Thay vào đó, để run_rag_pipeline tự raise đúng loại lỗi cụ thể
    # (LLMServiceUnavailable / VectorDBUnavailable / RequestTimeout từ
    # app/core/exceptions.py) khi biết chính xác lỗi đến từ đâu bên trong nó.
    # Các lỗi đó sẽ tự bay lên và được app_error_handler (đăng ký trong main.py)
    # xử lý, trả về đúng error_code + status_code tương ứng theo mục 30.
    #
    # Nếu là lỗi không lường trước (bug thật), để nó bay lên
    # unhandled_exception_handler -> trả "INTERNAL_ERROR", đã có log traceback
    # đầy đủ ở logger.exception(), không cần đoán/gán nhãn thay ở đây.
    answer, raw_sources = run_rag_pipeline(
        session_id=request.session_id,
        question=request.question,
        code=request.code,
        log=request.log,
    )

    # CHUẨN HÓA DỮ LIỆU NGUỒN (SOURCES)
    formatted_sources = [
        SourceItem(**src) if isinstance(src, dict) else src
        for src in raw_sources
    ]

    return ChatResponse(
        answer=answer,
        sources=formatted_sources,
        session_id=request.session_id,
    )
# #app/config.py — chứa MAX_QUESTION_LENGTH, MAX_CODE_LENGTH, MAX_LOG_LENGTH
# app/core/exceptions.py — chứa InvalidRequest, LLMServiceUnavailable, VectorDBUnavailable, RequestTimeout
# main.py phải gọi register_exception_handlers(app) để các lỗi trên được xử lý đúng thay vì rơi vào lỗi 500 mặc định.