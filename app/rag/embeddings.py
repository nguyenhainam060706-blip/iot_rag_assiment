# Cài đặt bằng: pip install langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings
from functools import lru_cache
import torch

@lru_cache(maxsize=1)
def get_embeddings():
    """
    Khởi tạo và trả về Embedding Model.
    Sử dụng @lru_cache để đảm bảo model chỉ được load 1 lần duy nhất vào bộ nhớ,
    các lần gọi sau sẽ tái sử dụng instance này.
    """
    print("Loading BGE-M3 Embeddings...")
    
    # Tự động phát hiện xem máy có cấu hình CUDA (GTX 1650) chưa, nếu chưa thì dùng CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding model đang chạy trên: {device.upper()}")
    
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True} # Quan trọng: Tăng độ chính xác khi so sánh similarity
    )