# Nên đổi import để tránh cảnh báo Deprecation trong tương lai
from langchain_chroma import Chroma 
from app.rag.embeddings import get_embeddings

def get_retriever(device_filter: str = None):
    print("Mở cửa nhà kho ChromaDB...")
    
    # 1. Gọi "Đôi mắt" BGE-M3
    embeddings = get_embeddings()
    
    # 2. Kết nối vào thư mục cơ sở dữ liệu
    vector_db = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings,
        collection_name="iot_knowledge" 
    )
    
    # 3. Cấu hình linh hoạt search_kwargs
    search_kwargs = {
        "k": 5, # Lấy 5 chunks
        "fetch_k": 20 # (Dành cho MMR) Lấy trước 20 chunks, sau đó chọn ra 5 chunks đa dạng nhất
    }
    
    # Nếu có truyền tên thiết bị, thêm filter vào cấu hình tìm kiếm
    if device_filter:
        search_kwargs["filter"] = {"device": device_filter}

    # 4. Cấu hình Thủ thư với thuật toán MMR
    retriever = vector_db.as_retriever(
        search_type="mmr", # Đổi từ similarity sang mmr
        search_kwargs=search_kwargs 
    )
    
    return retriever