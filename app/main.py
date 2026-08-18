from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.chat import router as chat_router

app = FastAPI(
    title="IoT Practical Assistant API",
    description="RAG Chatbot hỗ trợ sinh viên thực hành IoT",
    version="1.0.0"
)

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
        "llm": "qwen:4b", # Theo đúng spec của em
        "vector_db": "connected"
    }