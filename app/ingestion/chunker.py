import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Tính toán tỷ lệ quy đổi (Heuristic): 1 token tiếng Anh/Code ~ 4 ký tự
# Để đạt 500 - 1000 tokens, ta set chunk_size ở khoảng 2500 ký tự
CHUNK_SIZE = 2500 
CHUNK_OVERLAP = 500 

def extract_section_title(text: str) -> str:
    """
    (Heuristic) Cố gắng trích xuất tiêu đề mục (Section title) từ đầu đoạn text.
    Ví dụ tìm các mẫu: "5. MQTT", "5.1 CONNECT", "A. Lab 1", hoặc các dòng in hoa toàn bộ.
    Phục vụ cho mục tiêu "Section-aware chunking" (Section 12).
    """
    lines = text.split('\n')
    if not lines:
        return "Unknown Section"
        
    first_line = lines[0].strip()
    
    # Pattern tìm các tiêu đề đánh số kiểu "1.", "1.2", "A.", "2.1.3" ở đầu dòng
    section_pattern = r'^([A-Z0-9]+\.([0-9]+\.)*\s+[A-Za-z]+)'
    match = re.search(section_pattern, first_line)
    
    if match:
        return match.group(0) # Trả về phần tiêu đề đánh số
    elif first_line.isupper() and len(first_line) < 50:
        return first_line # Trả về dòng in hoa ngắn (thường là tiêu đề lớn)
        
    return ""

def chunk_documents(parsed_documents: List[Document]) -> List[Document]:
    """
    Chia nhỏ danh sách Document thành các chunk có kích thước phù hợp.
    """
    print(f"Bắt đầu Chunking {len(parsed_documents)} trang tài liệu...")
    
    # Sử dụng RecursiveCharacterTextSplitter
    # Nó sẽ cố gắng cắt ở "\n\n" (giữa các đoạn) trước, nếu vẫn dài thì cắt ở "\n" (giữa các dòng),
    # giúp giữ cấu trúc tự nhiên của văn bản và code tốt nhất có thể.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n\n", "\n\n", "\n", ".", " ", ""], 
        length_function=len
    )
    
    # LangChain tự động chia cắt và kế thừa metadata từ Document gốc (như source, page)
    chunks = text_splitter.split_documents(parsed_documents)
    
    # Bước bổ sung: Làm giàu metadata với Section Title (Section 12)
    current_section = "Unknown Section"
    
    for chunk in chunks:
        # Cố gắng tìm tiêu đề mới trong chunk này
        detected_section = extract_section_title(chunk.page_content)
        
        # Nếu tìm thấy một tiêu đề rõ ràng, cập nhật current_section
        if detected_section:
            current_section = detected_section
            
        # Gắn section vào metadata của chunk
        chunk.metadata["section"] = current_section
        
        # Đánh dấu chunk_id (hữu ích cho debug)
        chunk.metadata["chunk_length"] = len(chunk.page_content)
        
    print(f"Chunking hoàn tất. Tạo ra {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    # Test thử logic chunking và nhận diện section
    test_docs = [
        Document(
            page_content="5. MQTT Protocol\nMQTT là giao thức nhắn tin pub/sub...\n\n" * 20 + 
                         "5.1 CONNECT\nBản tin CONNECT dùng để thiết lập kết nối...\n" * 20,
            metadata={"source": "MQTT_Lab.pdf", "page": 12}
        )
    ]
    
    result_chunks = chunk_documents(test_docs)
    
    for i, chunk in enumerate(result_chunks[:2]):
        print(f"\n--- Chunk {i+1} ---")
        print("Metadata:", chunk.metadata)
        print("Content (snippet):", chunk.page_content[:100], "...")