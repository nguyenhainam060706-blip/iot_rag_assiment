import os
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from app.config import settings
from app.core.logging_middleware import logger
# Ngưỡng ký tự tối thiểu/trang để coi là "có nội dung text thật".
# Dưới ngưỡng này nhiều khả năng là scanned PDF (ảnh, không có text layer)
# -> PyMuPDFLoader vẫn "load thành công" nhưng page_content gần như rỗng,
# không hề raise lỗi, nên phải tự kiểm tra thủ công (mục 11.2).
MIN_CHARS_PER_PAGE_THRESHOLD = 20
def load_single_pdf(file_path: str) -> List[Document]:
    """
    Sử dụng PyMuPDFLoader (theo Section 11.1) để đọc một file PDF duy nhất.
    PyMuPDF giữ cấu trúc text tốt hơn so với pypdf, đặc biệt hữu ích khi xử lý
    datasheet chứa nhiều bảng biểu và format đặc thù.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    try:
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
    except Exception as e:
        logger.error(f"[INGESTION] Lỗi khi đọc file {file_path}: {e}")
        return []

    # --- Kiểm tra scanned PDF / PDF không có text layer (mục 11.2) ---
    # Không tự động OCR ở đây (OCR là bước riêng, chưa implement), nhưng
    # PHẢI cảnh báo rõ ràng để không âm thầm mất dữ liệu mà không ai biết.
    total_chars = sum(len(doc.page_content.strip()) for doc in docs)
    avg_chars_per_page = total_chars / len(docs) if docs else 0

    if docs and avg_chars_per_page < MIN_CHARS_PER_PAGE_THRESHOLD:
        logger.warning(
            f"[INGESTION] File '{file_path}' có vẻ là scanned PDF hoặc không "
            f"có text layer (trung bình {avg_chars_per_page:.1f} ký tự/trang). "
            f"Cần chạy qua bước OCR (mục 11.2) trước khi ingest, nếu không "
            f"nội dung file này sẽ KHÔNG được đưa vào Knowledge Base."
        )

    return docs


def load_directory(directory_path: str) -> List[Document]:
    """
    Quét toàn bộ thư mục (bao gồm cả thư mục con) để tìm và load tất cả các file PDF.
    Phục vụ cho cấu trúc Data Directory theo Section 9 của Tech Spec.
    """
    if not os.path.exists(directory_path):
        logger.warning(f"[INGESTION] Thư mục {directory_path} không tồn tại.")
        return []

    all_docs = []

    # Duyệt đệ quy qua các thư mục con
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(root, file)
                logger.info(f"[INGESTION] Đang đọc tài liệu: {file_path}...")
                docs = load_single_pdf(file_path)

                # Bổ sung một số metadata mặc định cấp file (sẽ được làm giàu thêm ở bước chunking)
                for doc in docs:
                    # Lấy tên thư mục cha (ví dụ: 'esp32', 'mqtt') để phân loại nội dung
                    category = os.path.basename(root)
                    doc.metadata["category_folder"] = category

                all_docs.extend(docs)

    logger.info(f"[INGESTION] Tổng cộng đã đọc {len(all_docs)} trang tài liệu từ {directory_path}")
    return all_docs


def load_knowledge_base(base_dir: str = None) -> List[Document]:
    """
    Hàm tổng hợp để load toàn bộ Knowledge Base theo cấu trúc quy định tại Section 9.

    LƯU Ý QUAN TRỌNG: bao gồm cả thư mục "raw_uploads" -- đây là nơi
    documents.py (API upload) lưu file PDF vào. Nếu bỏ sót thư mục này,
    mọi file sinh viên/admin upload qua POST /api/documents sẽ KHÔNG BAO
    GIỜ được ingest vào ChromaDB, dù nằm đúng vị trí trên đĩa.
    """
    if base_dir is None:
        # Dùng config thay vì hardcode "knowledge" để không phụ thuộc vào
        # việc script được chạy từ thư mục nào (cwd khác nhau tùy cách gọi).
        base_dir = settings.KNOWLEDGE_BASE_DIR

    # Các thư mục chính theo cấu trúc mục 9 + "raw_uploads" (nơi API upload lưu file)
    target_dirs = [
        "kit", "protocols", "experiments", "datasheets",
        "troubleshooting", "faq", "raw_uploads",
    ]

    knowledge_docs = []

    for dir_name in target_dirs:
        full_path = os.path.join(base_dir, dir_name)
        if os.path.exists(full_path):
            logger.info(f"[INGESTION] --- Đang nạp danh mục: {dir_name.upper()} ---")
            docs = load_directory(full_path)
            knowledge_docs.extend(docs)

    return knowledge_docs


if __name__ == "__main__":
    # Test thử quá trình đọc (cần tạo sẵn thư mục knowledge/ và nhét vài file pdf vào để test)
    # Ví dụ: knowledge/datasheets/mcu/ESP32_Datasheet.pdf
    print("Bắt đầu khởi động Document Loader...")
    documents = load_knowledge_base()
    if documents:
        print(f"\nLoad thành công. Đã nạp được {len(documents)} trang.")
        print("Ví dụ metadata trang đầu tiên:", documents[0].metadata)
    else:
        print("Không tìm thấy tài liệu nào. Hãy kiểm tra lại thư mục.")