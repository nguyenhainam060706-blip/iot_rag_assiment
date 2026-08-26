import torch
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings

# Import config và logger chung của toàn hệ thống
from app.config import settings
from app.core.logging_middleware import logger

@lru_cache(maxsize=1)
def get_embeddings():
    """
    Khởi tạo và trả về Embedding Model.
    Sử dụng @lru_cache để đảm bảo model chỉ được load 1 lần duy nhất vào bộ nhớ (Singleton),
    giúp chia sẻ chung 1 instance BGE-M3 giữa API Chat (pipeline.py) và API Ingestion,
    ngăn chặn triệt để lỗi Out of Memory (OOM) trên VRAM 4GB.
    """
    try:
        logger.info(f"[EMBEDDINGS] Bắt đầu load model: {settings.EMBEDDING_MODEL}...")
        
        # Tự động phát hiện xem máy có cấu hình CUDA (GTX 1650) chưa, nếu chưa thì dùng CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[EMBEDDINGS] Model đang được cấu hình chạy trên thiết bị: {device.upper()}")
        
        # Trả về instance của HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,  # Lấy linh động từ cấu hình (.env)
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True} # Quan trọng: Tăng độ chính xác khi so sánh similarity
        )
    except Exception as e:
        # Ghi log lỗi nếu không tải được model (do sai tên typo, hoặc mất mạng khi tải lần đầu)
        logger.error(f"[EMBEDDINGS] Lỗi nghiêm trọng khi khởi tạo Embedding Model: {str(e)}", exc_info=True)
        raise e