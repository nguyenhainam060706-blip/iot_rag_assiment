import asyncio
import httpx
from fastapi import APIRouter
from pydantic import BaseModel
from app.config import settings
# Khởi tạo ChromaDB client ở module-level để tránh overhead I/O mỗi request
try:
    import chromadb
    chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
except Exception:
    chroma_client = None
router = APIRouter()
class HealthResponse(BaseModel):
    status: str          # "ok" | "degraded"
    llm: str             # tên model, hoặc "<model> (unreachable)"
    vector_db: str       # "connected" | "disconnected"
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    API Health Check (Theo Section 26.1)
    Kiểm tra thực tế trạng thái LLM và Vector DB mà không làm block event loop.
    """
    
    # --- 1. Kiểm tra ChromaDB ---
    vector_db_status = "connected"
    if chroma_client is None:
        vector_db_status = "disconnected"
    else:
        try:
            # Đưa thao tác đồng bộ (blocking I/O) vào threadpool 
            # để bảo vệ event loop chính của FastAPI
            await asyncio.to_thread(chroma_client.heartbeat)
        except Exception:
            vector_db_status = "disconnected"

    # --- 2. Kiểm tra Ollama ---
    # Giới hạn timeout 0.8s để đảm bảo API luôn phản hồi < 1s
    health_check_timeout = min(getattr(settings, 'LLM_TIMEOUT_SECONDS', 2.0), 0.8)
    llm_status = settings.LLM_MODEL
    
    try:
        async with httpx.AsyncClient(timeout=health_check_timeout) as client:
            resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if resp.status_code != 200:
                llm_status = f"{settings.LLM_MODEL} (unreachable)"
    except Exception:
        llm_status = f"{settings.LLM_MODEL} (unreachable)"

    # --- 3. Tổng hợp trạng thái ---
    overall_status = (
        "ok"
        if vector_db_status == "connected" and "unreachable" not in llm_status
        else "degraded"
    )

    return HealthResponse(
        status=overall_status,
        llm=llm_status,
        vector_db=vector_db_status,
    )