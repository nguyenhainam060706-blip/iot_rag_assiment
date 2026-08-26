import re
from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
CHUNK_SIZE = 2500 
CHUNK_OVERLAP = 500 

def extract_section_title(text: str) -> str:
    """
    Trích xuất tiêu đề mục (Section title) từ toàn bộ text của chunk.
    Sẽ trả về tiêu đề hợp lệ XUẤT HIỆN CUỐI CÙNG trong text để đảm bảo 
    các chunk tiếp theo kế thừa đúng ngữ cảnh (nếu tiêu đề nằm giữa chunk).
    """
    lines = text.split('\n')
    last_found_section = ""
    
    # Pattern: Bắt đầu bằng chữ/số, hỗ trợ .số (1.2.3), có/không có dấu chấm, khoảng trắng, chữ cái
    section_pattern = r'^([A-Z]|\d+)(?:\.\d+)*\.?\s+[A-ZÀ-Ỹa-zà-ỹ].*'
    
    for line in lines:
        clean_line = line.strip()
        
        # Bỏ qua các dòng trống hoặc quá dài (tiêu đề thường ngắn gọn)
        if not clean_line or len(clean_line) > 100:
            continue
            
        # 1. Kiểm tra bằng Regex cho tiêu đề đánh số
        if re.match(section_pattern, clean_line):
            last_found_section = clean_line
            continue
            
        # 2. Heuristic chữ HOA toàn bộ
        if clean_line.isupper() and len(clean_line) < 60 and ":" not in clean_line:
            # Ngăn chặn nhận nhầm mã định danh (GPIO34, ESP_ERR, MAX485) thành tiêu đề.
            words = clean_line.split()
            # Chấp nhận nếu có >= 2 từ, HOẶC nếu 1 từ thì phải là thuần chữ (vd: INTRODUCTION)
            if len(words) > 1 or words[0].isalpha():
                last_found_section = clean_line
            
    return last_found_section

def chunk_documents(parsed_documents: List[Document]) -> List[Document]:
    """
    Chia nhỏ danh sách Document thành các chunk, đảm bảo cô lập metadata theo từng file nguồn.
    """
    print(f"Bắt đầu Chunking {len(parsed_documents)} trang tài liệu...")
    
    # Không dùng "." làm separator để tránh vỡ các thông số (192.168.1.1, 3.3V, 5.1.2)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n\n", "\n\n", "\n", " ", ""], 
        length_function=len
    )
    
    # Gom nhóm tài liệu theo nguồn (source) trước khi xử lý để tránh rò rỉ (leak) section
    docs_by_source: Dict[str, List[Document]] = {}
    for doc in parsed_documents:
        source = doc.metadata.get("source", "Unknown_Source")
        if source not in docs_by_source:
            docs_by_source[source] = []
        docs_by_source[source].append(doc)
        
    final_chunks = []
    
    # Xử lý tuần tự từng file nguồn
    for source, docs in docs_by_source.items():
        # Cắt chunk cho toàn bộ các trang thuộc file này
        source_chunks = text_splitter.split_documents(docs)
        
        # Reset current_section mỗi khi bắt đầu một file nguồn MỚI
        current_section = "Unknown Section"
        
        for chunk in source_chunks:
            # Structure Detection hoạt động trên toàn bộ chunk (quét tất cả các dòng)
            detected_section = extract_section_title(chunk.page_content)
            
            # Nếu phát hiện section mới, cập nhật state
            if detected_section:
                current_section = detected_section
                
            # Gắn metadata
            chunk.metadata["section"] = current_section
            chunk.metadata["chunk_length"] = len(chunk.page_content)
            
            final_chunks.append(chunk)
            
    print(f"Chunking hoàn tất. Tạo ra {len(final_chunks)} chunks.")
    return final_chunks

if __name__ == "__main__":
    # Test thử với các case phức tạp: Tiêu đề nằm giữa chunk, tiếng Việt, số liệu kỹ thuật
    test_docs = [
        Document(
            page_content="Nội dung trang trước kéo dài...\n5.1 Kết nối MQTT\nBản tin CONNECT dùng để thiết lập kết nối...",
            metadata={"source": "MQTT_Lab.pdf", "page": 12}
        ),
        Document(
            page_content="INTRODUCTION\nĐây là phần giới thiệu.\nGPIO34\nChân GPIO34 chỉ nhận dòng nhỏ. Địa chỉ IP router thường là 192.168.1.1\nESP_ERR_TIMEOUT\n9. Troubleshooting\nCác lỗi thường gặp...",
            metadata={"source": "ESP32_Datasheet.pdf", "page": 5}
        )
    ]
    
    result_chunks = chunk_documents(test_docs)
    
    for i, chunk in enumerate(result_chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Nguồn: {chunk.metadata.get('source')} | Mục: {chunk.metadata.get('section')}")
        print("Trích xuất text (snippet):", chunk.page_content[:100].replace("\n", "\\n"), "...")