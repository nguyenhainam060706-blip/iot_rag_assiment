import os
os.environ["USE_TF"] = "0"
from typing import List
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
    LLM_TIMEOUT_SECONDS: float = 60.0 # Timeout khi gọi Ollama generate, tránh treo request vô hạn

    # --- 3. Cấu hình Embeddings & Vector DB ---
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION: str = "iot_knowledge"
    TOP_K_DEFAULT: int = 5 # Baseline theo mục 16 spec, cần benchmark lại K=3/5/10

    # --- 4. Cấu hình Đường dẫn Thư mục (Paths) ---
    # Sử dụng đường dẫn tương đối từ thư mục gốc của project
    RAW_DATA_DIR: str = "./data/raw"
    PROCESSED_DATA_DIR: str = "./data/processed"
    KNOWLEDGE_BASE_DIR: str = "./knowledge"

    # --- 5. Cấu hình CORS (mục 33 - Security) ---
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:5173,http://localhost:8080"

    @property
    def allowed_origins_list(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    # --- 6. Giới hạn độ dài input (mục 33 - Security) ---
    MAX_QUESTION_LENGTH: int = 2000
    MAX_CODE_LENGTH: int = 20000
    MAX_LOG_LENGTH: int = 20000

    # --- 7. Cấu hình Logging (mục 32) ---
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "./logs"

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
        settings.CHROMA_PERSIST_DIR,
        settings.LOG_DIR,
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Gọi hàm khởi tạo thư mục ngay khi config được load
init_directories()