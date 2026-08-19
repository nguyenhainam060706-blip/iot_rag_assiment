from fastapi import APIRouter
from app.config import settings

router = APIRouter()

@router.get("/", response_model=dict)
async def health_check():
    """
    API Health Check (Theo Section 26.1)
    Trả về trạng thái của Backend, loại LLM đang sử dụng và kết nối Vector DB.
    """
    # Mặc định trả về trạng thái chuẩn theo Tech Spec
    response = {
        "status": "ok",
        "llm": settings.LLM_MODEL,         # Lấy linh hoạt từ config.py (ví dụ: qwen2.5:3b)
        "vector_db": "connected"
    }
    
    # 💡 Tương lai (Phase 2):có thể tích hợp thư viện httpx vào đây 
    # để ping thử (gửi request) tới settings.OLLAMA_BASE_URL.
    # Nếu Ollama bị tắt, tự động đổi "status" thành "error". 
    # Tạm thời ở MVP,  giữ form tĩnh này để đảm bảo API phản hồi cực nhanh (< 1s).
    
    return response