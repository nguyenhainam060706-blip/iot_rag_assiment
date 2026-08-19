from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.chat import router as chat_router
from app.api import chat, documents # import thêm documents
app = FastAPI(
    title="IoT Practical Assistant API",
    description="RAG Chatbot hỗ trợ sinh viên thực hành IoT",
    version="1.0.0"
)
# Đăng ký router chat
app.include_router(chat.router, prefix="/api", tags=["Chat"])

# Đăng ký router quản lý tài liệu
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
# Cho phép Web Lab gọi API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Khi deploy thật, đổi thành IP của web
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gắn các đường dẫn API vào
app.include_router(chat_router, prefix="/api")

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "llm": "qwen:4b", # Theo đúng spec
        "vector_db": "connected"
    }