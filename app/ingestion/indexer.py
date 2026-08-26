import time
import hashlib
import threading
from typing import List
from langchain_chroma import Chroma

# Import cấu hình và hàm tạo Embedding chung của hệ thống
from app.config import settings
from app.rag.embeddings import get_embeddings
from app.core.logging_middleware import logger 

from app.ingestion.loader import load_knowledge_base
from app.ingestion.parser import parse_documents
from app.ingestion.chunker import chunk_documents

# Khóa (Lock) chống chạy trùng: Ngăn chặn 2 request /reindex chạy cùng lúc
_ingestion_lock = threading.Lock()

# Ruleset phân loại giúp dễ dàng mở rộng sau này
DOC_TYPE_RULES = {
    "datasheet": ["datasheet"],
    "lab_manual": ["lab", "experiments", "experiment"],
    "protocol_doc": ["protocol", "mqtt", "http"]
}

DEVICE_RULES = {
    "ESP32": ["esp32", "esp8266"],
    "MAX485": ["max485", "rs485"],
    "PZEM-004T": ["pzem", "pzem-004t"],
    "DHT11": ["dht11", "dht22"]
}

def _generate_deterministic_id(chunk) -> str:
    """
    Tạo ID tất định (Deterministic ID) dựa trên NỘI DUNG THỰC TẾ của chunk.
    Loại bỏ hoàn toàn sự phụ thuộc vào Index toàn cục. Dù file có bị dịch chuyển 
    thứ tự, ID của các chunk không đổi sẽ luôn giữ nguyên.
    """
    source = chunk.metadata.get("source", "unknown")
    page = str(chunk.metadata.get("page", 0))
    section = chunk.metadata.get("section", "unknown")
    
    # Sử dụng TOÀN BỘ nội dung chunk làm nguyên liệu hash.
    # Khẳng định tính duy nhất: 2 chunk cùng file, cùng page, cùng nội dung = 1 ID.
    content = chunk.page_content
    
    raw_id = f"{source}_{page}_{section}_{content}"
    return hashlib.md5(raw_id.encode('utf-8')).hexdigest()

def enrich_metadata(chunks):
    """
    Làm giàu và chuẩn hóa Metadata. 
    Hỗ trợ gán nhiều Device cho một tài liệu thay vì chỉ 1.
    """
    for chunk in chunks:
        source_lower = chunk.metadata.get("source", "").lower()
        folder_lower = chunk.metadata.get("category_folder", "").lower()
        context_str = f"{source_lower} {folder_lower}"
        
        # 1. Phân loại Document Type (Giữ nguyên dừng ở match đầu tiên vì 1 file thường chỉ thuộc 1 loại)
        chunk.metadata["document_type"] = "general" 
        for doc_type, keywords in DOC_TYPE_RULES.items():
            if any(kw in context_str for kw in keywords):
                chunk.metadata["document_type"] = doc_type
                break
                
        # 2. Phân loại Device (Lấy TẤT CẢ các thiết bị match được)
        matched_devices = []
        for device, keywords in DEVICE_RULES.items():
            if any(kw in context_str for kw in keywords):
                matched_devices.append(device)
                
        if matched_devices:
            # Nếu match nhiều thiết bị -> "ESP32, MAX485"
            chunk.metadata["device"] = ", ".join(matched_devices)
        else:
            chunk.metadata["device"] = "Unknown"
                
        # 3. Xử lý giá trị None (ChromaDB sẽ báo lỗi nếu metadata chứa giá trị None)
        for key, value in chunk.metadata.items():
            if value is None:
                chunk.metadata[key] = "Unknown"
                
    return chunks

def run_ingestion_pipeline(base_dir: str = None):
    """
    Thực thi toàn bộ Data Pipeline. Được bọc trong Try/Catch và Thread Lock
    để an toàn tuyệt đối khi làm Background Task.
    """
    if base_dir is None:
        base_dir = settings.KNOWLEDGE_BASE_DIR

    # KIỂM TRA KHÓA (MUTEX): Từ chối ngay nếu đang có tiến trình chạy ngầm
    if not _ingestion_lock.acquire(blocking=False):
        logger.warning("[INGESTION] Từ chối thực thi: Hệ thống đang có một tiến trình index chạy ngầm.")
        return False
        
    try:
        logger.info("[INGESTION] 🚀 BẮT ĐẦU CHẠY INGESTION PIPELINE...")
        start_time = time.time()
        
        # --- BƯỚC 1: Đọc PDF Thô ---
        logger.info("[INGESTION] [1/5] Đang đọc tài liệu từ ổ cứng...")
        raw_docs = load_knowledge_base(base_dir)
        if not raw_docs:
            logger.warning("[INGESTION] Không tìm thấy tài liệu nào trong thư mục.")
            return False
            
        # --- BƯỚC 2: Làm sạch Text ---
        logger.info("[INGESTION] [2/5] Đang làm sạch và nhận diện cấu trúc...")
        parsed_docs = parse_documents(raw_docs)
        
        # --- BƯỚC 3: Cắt thành Chunk ---
        logger.info("[INGESTION] [3/5] Đang chia nhỏ tài liệu (Chunking)...")
        chunks = chunk_documents(parsed_docs)
        
        # --- BƯỚC 4: Gắn Metadata ---
        logger.info("[INGESTION] [4/5] Đang làm giàu Metadata...")
        enriched_chunks = enrich_metadata(chunks)
        
        # --- BƯỚC 5: Nhúng (Embedding) & Lưu ChromaDB (XỬ LÝ BATCHING & VRAM) ---
        logger.info(f"[INGESTION] [5/5] Khởi tạo BGE-M3 & ChromaDB ({settings.CHROMA_PERSIST_DIR})...")
        
        embeddings = get_embeddings()
        
        vector_db = Chroma(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=settings.CHROMA_COLLECTION
        )
        
        # Sinh danh sách Deterministic ID ĐỘC LẬP VỚI INDEX TOÀN CỤC
        chunk_ids = [_generate_deterministic_id(chunk) for chunk in enriched_chunks]
        
        # CHIA BATCH ĐỂ BẢO VỆ VRAM 4GB
        BATCH_SIZE = 50
        total_chunks = len(enriched_chunks)
        logger.info(f"[INGESTION] Bắt đầu Insert/Upsert {total_chunks} chunks (Batch size: {BATCH_SIZE})...")
        
        for i in range(0, total_chunks, BATCH_SIZE):
            batch_chunks = enriched_chunks[i : i + BATCH_SIZE]
            batch_ids = chunk_ids[i : i + BATCH_SIZE]
            
            vector_db.add_documents(documents=batch_chunks, ids=batch_ids)
            logger.info(f"[INGESTION] Tiến độ: Đã lưu {min(i + BATCH_SIZE, total_chunks)}/{total_chunks} chunks.")
            
        end_time = time.time()
        logger.info(f"[INGESTION] ✅ HOÀN TẤT! Đã đồng bộ {total_chunks} chunks vào Vector DB.")
        logger.info(f"[INGESTION] ⏱️ Tổng thời gian chạy: {round(end_time - start_time, 2)} giây.")
        
        return True
        
    except Exception as e:
        logger.error(f"[INGESTION] LỖI NGHIÊM TRỌNG TRONG QUÁ TRÌNH PIPELINE: {str(e)}", exc_info=True)
        return False
        
    finally:
        _ingestion_lock.release()

if __name__ == "__main__":
    run_ingestion_pipeline()