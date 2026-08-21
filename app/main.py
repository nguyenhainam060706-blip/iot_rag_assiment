from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# Import các router từ thư mục api
from app.api import chat, documents, health

app = FastAPI(
    title="IoT Practical Assistant API",
    description="RAG Chatbot hỗ trợ sinh viên thực hành IoT",
    version="1.0.0"
)

# 1. CẤU HÌNH CORS (Đã fix lỗi bảo mật)
ALLOWED_ORIGINS = [
    "http://localhost:3000",   
    "http://127.0.0.1:5173",   
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS, 
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. ĐĂNG KÝ ROUTER
# Gọi router health (từ health.py)
app.include_router(health.router, prefix="/api", tags=["System"])

# Gọi router chat và documents
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])