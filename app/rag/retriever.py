from langchain_community.vectorstores import Chroma
from app.rag.embeddings import get_embeddings

def get_retriever():
    print("Mở cửa nhà kho ChromaDB...")
    
    # 1. Gọi "Đôi mắt" BGE-M3 để biến câu hỏi của sinh viên thành Vector
    embeddings = get_embeddings()
    
    # 2. Kết nối vào thư mục cơ sở dữ liệu đã được nạp từ trước
    vector_db = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings,
        collection_name="iot_knowledge" # Tên kho tài liệu (Section 15.2)
    )
    
    # 3. Cấu hình Thủ thư (Retriever)
    # search_type="similarity": Tìm kiếm dựa trên độ tương đồng ý nghĩa
    # k=5: Rút ra đúng 5 đoạn tài liệu liên quan nhất
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5} 
    )
    
    return retriever