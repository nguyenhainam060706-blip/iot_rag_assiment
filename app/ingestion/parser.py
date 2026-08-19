import re
from typing import List
from langchain_core.documents import Document

def clean_text(text: str) -> str:
    """
    Dọn dẹp các ký tự thừa thãi, khoảng trắng không cần thiết do lỗi OCR 
    hoặc do PyMuPDF bóc tách từ các định dạng layout phức tạp.
    """
    if not text:
        return ""
    
    # Xóa khoảng trắng thừa liên tiếp, giữ lại xuống dòng (\n)
    cleaned = re.sub(r'[ \t]+', ' ', text)
    
    # Sửa lỗi một số ký tự đặc biệt thường gặp trong datasheet
    cleaned = cleaned.replace("ﬁ", "fi").replace("ﬂ", "fl")
    cleaned = cleaned.replace("©", "(c)").replace("®", "(R)")
    
    return cleaned.strip()

def remove_header_footer(text: str) -> str:
    """
    Loại bỏ các đoạn text lặp đi lặp lại ở đầu và cuối mỗi trang PDF.
    Ví dụ: Số trang, Tên tài liệu, "Espressif Systems", "Confidential".
    """
    # Tách text thành các dòng
    lines = text.split('\n')
    
    if len(lines) < 3:
        return text
        
    # Heuristic đơn giản: Xóa 1-2 dòng đầu và 1-2 dòng cuối nếu chúng ngắn 
    # và chứa các từ khóa đặc trưng
    header_keywords = ["espressif", "datasheet", "user manual", "revision"]
    footer_keywords = ["page", "confidential", "copyright"]
    
    # Kiểm tra dòng đầu (Header)
    if any(keyword in lines[0].lower() for keyword in header_keywords) or len(lines[0]) < 20:
        lines = lines[1:]
        
    # Kiểm tra dòng cuối (Footer)
    if lines:
        if any(keyword in lines[-1].lower() for keyword in footer_keywords) or re.search(r'\b\d+\b', lines[-1]) and len(lines[-1]) < 10:
            lines = lines[:-1]
            
    return '\n'.join(lines)

def detect_code_blocks(text: str) -> str:
    """
    (Heuristic) Cố gắng nhận diện các đoạn text có vẻ giống code C/C++ 
    và bọc nó lại bằng markdown formatting (```c ... ```) để LLM sau này dễ hiểu hơn.
    Phục vụ trực tiếp cho mục tiêu Code Analysis (Section 23).
    """
    # Nếu trong text có các pattern rõ ràng của code C/Arduino
    code_patterns = [
        r'#include\s+<.*?>',
        r'void\s+setup\(\)',
        r'void\s+loop\(\)',
        r'int\s+main\(\)',
        r'esp_err_t'
    ]
    
    # Nếu tìm thấy một trong các pattern trên, ta bọc toàn bộ đoạn đó lại.
    # Lưu ý: Đây là một xử lý đơn giản. Trong thực tế, nếu PDF có đánh dấu format riêng
    # thì có thể bóc tách chính xác hơn.
    is_code = any(re.search(pattern, text) for pattern in code_patterns)
    
    if is_code:
        # Bọc lại bằng markdown (Giả định đoạn text đang xét chủ yếu là code)
        return f"```c\n{text}\n```"
    
    return text

def parse_documents(raw_documents: List[Document]) -> List[Document]:
    """
    Hàm xử lý chính (Pipeline con): Nhận vào danh sách Document thô từ loader,
    làm sạch, định dạng lại và trả về danh sách Document đã được chuẩn hóa.
    """
    parsed_docs = []
    
    print(f"Bắt đầu dọn dẹp (Parsing) {len(raw_documents)} trang tài liệu...")
    
    for doc in raw_documents:
        original_text = doc.page_content
        
        # 1. Dọn dẹp khoảng trắng
        processed_text = clean_text(original_text)
        
        # 2. Bỏ header/footer
        processed_text = remove_header_footer(processed_text)
        
        # 3. Đánh dấu khối code (Nếu có)
        processed_text = detect_code_blocks(processed_text)
        
        # Chỉ giữ lại những trang có nội dung sau khi dọn dẹp
        if len(processed_text.strip()) > 20: 
            # Tạo một Document mới với nội dung đã sửa, giữ nguyên metadata cũ
            new_doc = Document(
                page_content=processed_text,
                metadata=doc.metadata
            )
            
            # Gắn thêm một cờ vào metadata để biết đã đi qua bước parser
            new_doc.metadata["parsed"] = True
            parsed_docs.append(new_doc)
            
    print(f"Parsing hoàn tất. Giữ lại {len(parsed_docs)} trang hợp lệ.")
    return parsed_docs

if __name__ == "__main__":
    # Test thử nhanh logic parser
    test_text = """Espressif Systems V1.2
    
    #include <stdio.h>
    void app_main() {
        printf("Hello World\n");
    }
    
    Page 42"""
    
    doc = Document(page_content=test_text, metadata={"source": "test.pdf"})
    result = parse_documents([doc])
    
    print("Nội dung gốc:\n", doc.page_content)
    print("\n---")
    print("\nNội dung sau khi Parse:\n", result[0].page_content)