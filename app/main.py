import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import các router từ thư mục api
from app.api import chat, documents, health
from app.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging_middleware import RequestLoggingMiddleware, logger

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG_MODE,
)

# 1. CẤU HÌNH CORS
# Origin đọc từ .env qua config.py (mục 33: chỉ bind localhost/LAN, không hardcode "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. LOGGING MIDDLEWARE (mục 32)
app.add_middleware(RequestLoggingMiddleware)

# 3. EXCEPTION HANDLERS TẬP TRUNG (mục 30)
register_exception_handlers(app)

# 4. ĐĂNG KÝ ROUTER
# health.router tự chứa path "/health" -> GET /api/health
app.include_router(health.router, prefix="/api", tags=["System"])

# prefix="/api/chat" + route con là "" (không dấu "/") -> path cuối: POST /api/chat
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])

# prefix="/api/documents" + route con "" -> GET/POST /api/documents,
# DELETE /api/documents/{id}, POST /api/documents/reindex
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])


@app.on_event("startup")
async def startup_check():
    """
    Kiểm tra sớm (fail-fast) các dependency quan trọng: Ollama, ChromaDB.
    Không raise exception ở đây để tránh app không start được hoàn toàn
    khi lab đang setup dở dang — chỉ log cảnh báo rõ ràng cho người vận hành thấy.
    """
    logger.info("Starting %s v%s (debug=%s)", settings.APP_NAME, settings.APP_VERSION, settings.DEBUG_MODE)

    # --- Kiểm tra Ollama ---
    try:
        import httpx
        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
            resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if resp.status_code == 200:
                logger.info("Ollama OK tại %s (model=%s)", settings.OLLAMA_BASE_URL, settings.LLM_MODEL)
            else:
                logger.warning("Ollama phản hồi bất thường: %s", resp.status_code)
    except Exception as e:
        logger.warning("Không kết nối được Ollama tại %s: %s", settings.OLLAMA_BASE_URL, e)

    # --- Kiểm tra ChromaDB ---
    try:
        import chromadb
        client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        client.heartbeat()
        logger.info("ChromaDB OK tại %s (collection=%s)", settings.CHROMA_PERSIST_DIR, settings.CHROMA_COLLECTION)
    except Exception as e:
        logger.warning("Không khởi tạo được ChromaDB tại %s: %s", settings.CHROMA_PERSIST_DIR, e)

    # --- Cảnh báo nếu DEBUG_MODE bật khi có vẻ đang chạy thật (mục 33) ---
    if settings.DEBUG_MODE:
        logger.warning(
            "DEBUG_MODE=True — chỉ nên bật khi phát triển local. "
            "Nhớ set DEBUG_MODE=False trong .env trước khi chạy thật cho sinh viên dùng."
        )