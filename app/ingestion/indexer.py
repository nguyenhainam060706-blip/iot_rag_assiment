import time
from langchain_chroma import Chroma

# Import cấu hình và hàm tạo Embedding chung của hệ thống
from app.config import settings
from app.rag.embeddings import get_embeddings

# Import 3 công đoạn xử lý bạn đã viết trước đó
from app.ingestion.loader import load_knowledge_base
from app.ingestion.parser import parse_documents
from app.ingestion.chunker import chunk_documents

def enrich_metadata(chunks):
    """
    Làm giàu và chuẩn hóa Metadata theo Section 13 của Tech Spec.
    Bước này giúp AI sau này có thể lọc chính xác (ví dụ: chỉ tìm trong datasheet).
    """
    for chunk in chunks:
        # Lấy tên file gốc (ví dụ: ESP32_Datasheet.pdf)
        source = chunk.metadata.get("source", "").lower()
        folder = chunk.metadata.get("category_folder", "").lower()
        
        # 1. Phân loại Document Type
        if "datasheet" in source or "datasheets" in folder:
            chunk.metadata["document_type"] = "datasheet"
        elif "lab" in source or "experiments" in folder:
            chunk.metadata["document_type"] = "lab_manual"
        elif "protocol" in folder:
            chunk.metadata["document_type"] = "protocol_doc"
        else:
            chunk.metadata["document_type"] = "general"
            
        # 2. Phân loại Device (Heuristic)
        if "esp32" in source:
            chunk.metadata["device"] = "ESP32"
        elif "max485" in source or "rs485" in source:
            chunk.metadata["device"] = "MAX485"
            
        # 3. Xử lý giá trị None (ChromaDB sẽ báo lỗi nếu metadata chứa giá trị None)
        for key, value in chunk.metadata.items():
            if value is None:
                chunk.metadata[key] = "Unknown"
                
    return chunks

def run_ingestion_pipeline(base_dir: str = settings.KNOWLEDGE_BASE_DIR):
    """
    Thực thi toàn bộ Data Pipeline. Hàm này có thể được gọi ngầm (Background Task) 
    từ API /api/documents/reindex.
    """
    print("🚀 BẮT ĐẦU CHẠY INGESTION PIPELINE...")
    start_time = time.time()
    
    # --- BƯỚC 1: Đọc PDF Thô ---
    print("\n[1/5] Đang đọc tài liệu từ ổ cứng...")
    raw_docs = load_knowledge_base(base_dir)
    if not raw_docs:
        print(" Không tìm thấy tài liệu nào. Vui lòng ném file PDF vào thư mục knowledge/ và thử lại.")
        return False
        
    # --- BƯỚC 2: Làm sạch Text ---
    print("\n[2/5] Đang làm sạch và nhận diện cấu trúc...")
    parsed_docs = parse_documents(raw_docs)
    
    # --- BƯỚC 3: Cắt thành Chunk ---
    print("\n[3/5] Đang chia nhỏ tài liệu (Chunking)...")
    chunks = chunk_documents(parsed_docs)
    
    # --- BƯỚC 4: Gắn Metadata ---
    print("\n[4/5] Đang làm giàu Metadata...")
    enriched_chunks = enrich_metadata(chunks)
    
    # --- BƯỚC 5: Nhúng (Embedding) & Lưu ChromaDB ---
    print(f"\n[5/5] Đang khởi tạo BGE-M3 và lưu vào ChromaDB ({settings.CHROMA_PERSIST_DIR})...")
    # Quá trình này sẽ hơi lâu vì BGE-M3 phải biến hàng nghìn chữ thành Vector
    embeddings = get_embeddings()
    
    vector_db = Chroma(
        persist_directory=settings.CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=settings.CHROMA_COLLECTION
    )
    
    # Đẩy dữ liệu vào DB (Cập nhật ghi đè hoặc thêm mới)
    vector_db.add_documents(enriched_chunks)
    
    end_time = time.time()
    print(f"\n✅ INGESTION HOÀN TẤT! Đã đưa {len(enriched_chunks)} chunks vào Vector DB.")
    print(f"⏱️ Tổng thời gian chạy: {round(end_time - start_time, 2)} giây.")
    
    return True

if __name__ == "__main__":
    # Chạy thử trực tiếp file này
    run_ingestion_pipeline()