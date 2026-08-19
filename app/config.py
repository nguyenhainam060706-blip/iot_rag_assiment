import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # --- 1. Cấu hình Ứng dụng (App Settings) ---
    APP_NAME: str = "IoT Practical Assistant API"
    APP_VERSION: str = "1.0.0"
    DEBUG_MODE: bool = True
    
    # --- 2. Cấu hình LLM (Ollama) ---
    OLLAMA_BASE_URL: str = "http://localhost:11434" # URL mặc định của Ollama local
    LLM_MODEL: str = "qwen2.5:3b" # Có thể đổi thành qwen3:4b tùy theo phiên bản bạn pull
    LLM_TEMPERATURE: float = 0.0 # Bắt buộc set thấp (0.0) để AI không bịa datasheet (Hallucination)
    
    # --- 3. Cấu hình Embeddings & Vector DB ---
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION: str = "iot_knowledge"
    
    # --- 4. Cấu hình Đường dẫn Thư mục (Paths) ---
    # Sử dụng đường dẫn tương đối từ thư mục gốc của project
    RAW_DATA_DIR: str = "./data/raw"
    PROCESSED_DATA_DIR: str = "./data/processed"
    KNOWLEDGE_BASE_DIR: str = "./knowledge"
    
    class Config:
        # Pydantic sẽ tự động tìm và đọc file .env ở cùng thư mục chạy code
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Bỏ qua các cảnh báo nếu trong file .env có biến lạ không được định nghĩa ở đây
        extra = "ignore" 

# Khởi tạo một object settings duy nhất để import ở các file khác
settings = Settings()

# --- Tự động tạo cấu trúc thư mục nếu chưa có ---
def init_directories():
    directories = [
        settings.RAW_DATA_DIR, 
        settings.PROCESSED_DATA_DIR, 
        settings.KNOWLEDGE_BASE_DIR,
        settings.CHROMA_PERSIST_DIR
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Gọi hàm khởi tạo thư mục ngay khi config được load
init_directories()