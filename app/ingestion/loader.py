import os
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

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
        return docs
    except Exception as e:
        print(f"Lỗi khi đọc file {file_path}: {e}")
        return []

def load_directory(directory_path: str) -> List[Document]:
    """
    Quét toàn bộ thư mục (bao gồm cả thư mục con) để tìm và load tất cả các file PDF.
    Phục vụ cho cấu trúc Data Directory theo Section 9 của Tech Spec.
    """
    if not os.path.exists(directory_path):
        print(f"Cảnh báo: Thư mục {directory_path} không tồn tại.")
        return []

    all_docs = []
    
    # Duyệt đệ quy qua các thư mục con
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(root, file)
                print(f"Đang đọc tài liệu: {file_path}...")
                docs = load_single_pdf(file_path)
                
                # Bổ sung một số metadata mặc định cấp file (sẽ được làm giàu thêm ở bước chunking)
                for doc in docs:
                    # Lấy tên thư mục cha (ví dụ: 'esp32', 'mqtt') để phân loại nội dung
                    category = os.path.basename(root)
                    doc.metadata["category_folder"] = category
                    
                all_docs.extend(docs)
                
    print(f"Tổng cộng đã đọc {len(all_docs)} trang tài liệu từ {directory_path}")
    return all_docs

def load_knowledge_base(base_dir: str = "knowledge") -> List[Document]:
    """
    Hàm tổng hợp để load toàn bộ Knowledge Base theo cấu trúc quy định tại Section 9.
    """
    # Các thư mục chính cần quét
    target_dirs = ["kit", "protocols", "experiments", "datasheets", "troubleshooting", "faq"]
    
    knowledge_docs = []
    
    for dir_name in target_dirs:
        full_path = os.path.join(base_dir, dir_name)
        if os.path.exists(full_path):
            print(f"\n--- Đang nạp danh mục: {dir_name.upper()} ---")
            docs = load_directory(full_path)
            knowledge_docs.extend(docs)
            
    return knowledge_docs

if __name__ == "__main__":
    # Test thử quá trình đọc (cần tạo sẵn thư mục knowledge/ và nhét vài file pdf vào để test)
    # Ví dụ: knowledge/datasheets/mcu/ESP32_Datasheet.pdf
    print("Bắt đầu khởi động Document Loader...")
    documents = load_knowledge_base(base_dir="../../knowledge")
    if documents:
        print(f"\nLoad thành công. Đã nạp được {len(documents)} trang.")
        print("Ví dụ metadata trang đầu tiên:", documents[0].metadata)
    else:
        print("Không tìm thấy tài liệu nào. Hãy kiểm tra lại thư mục.")