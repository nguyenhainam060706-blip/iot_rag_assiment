import re
from typing import List
from langchain_core.documents import Document
from app.core.logging_middleware import logger

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
    [ĐÃ VÁ LỖI]: Bỏ điều kiện len < 20 để không làm mất các tiêu đề ngắn (vd: "5.1 MQTT").
    Chỉ xóa khi khớp chính xác với từ khóa header/footer hoặc là số trang đơn độc.
    """
    lines = text.split('\n')
    
    if len(lines) < 3:
        return text
        
    header_keywords = ["espressif systems", "confidential", "all rights reserved"]
    footer_keywords = ["page", "confidential", "copyright"]
    
    # Kiểm tra dòng đầu (Header): Phải chứa từ khóa rõ ràng, không dùng điều kiện độ dài nữa
    first_line_lower = lines[0].strip().lower()
    if any(keyword in first_line_lower for keyword in header_keywords):
        lines = lines[1:]
        
    # Kiểm tra dòng cuối (Footer): Chứa từ khóa footer HOẶC là dòng chỉ chứa số trang (vd: "Page 42" hoặc "42")
    if lines:
        last_line_lower = lines[-1].strip().lower()
        is_page_number_only = bool(re.match(r'^(page\s*)?\d+$', last_line_lower))
        
        if any(keyword in last_line_lower for keyword in footer_keywords) or is_page_number_only:
            lines = lines[:-1]
            
    return '\n'.join(lines)

def detect_code_blocks(text: str) -> str:
    """
    Nhận diện chính xác các đoạn code C/C++ bằng cách kiểm tra cú pháp đầu dòng 
    hoặc cấu trúc hàm đặc trưng, tránh bọc nhầm văn xuôi giải thích lý thuyết.
    """
    # Các pattern yêu cầu tính cấu trúc cao (bắt đầu dòng hoặc có cú pháp rõ ràng)
    strict_code_patterns = [
        r'^\s*#include\s+<.*?>',
        r'^\s*(void|int|bool|esp_err_t|static)\s+\w+\s*\(.*?\)\s*\{',
        r'^\s*esp_err_t\s+\w+\s*='
    ]
    
    # Kiểm tra xem có ít nhất một pattern xuất hiện như một dòng code thực thụ hay không
    is_code = any(re.search(pattern, text, re.MULTILINE) for pattern in strict_code_patterns)
    
    if is_code:
        return f"```c\n{text}\n```"
    
    return text

def parse_documents(raw_documents: List[Document]) -> List[Document]:
    """
    Hàm xử lý chính (Pipeline con): Nhận vào danh sách Document thô từ loader,
    làm sạch, định dạng lại và trả về danh sách Document đã được chuẩn hóa.
    """
    parsed_docs = []
    
    logger.info(f"[PARSER] Bắt đầu dọn dẹp (Parsing) {len(raw_documents)} trang tài liệu...")
    
    for doc in raw_documents:
        original_text = doc.page_content
        
        # 1. Dọn dẹp khoảng trắng
        processed_text = clean_text(original_text)
        
        # 2. Bỏ header/footer (Đã an toàn với tiêu đề ngắn)
        processed_text = remove_header_footer(processed_text)
        
        # 3. Đánh dấu khối code (Đã an toàn với văn xuôi)
        processed_text = detect_code_blocks(processed_text)
        
        # Chỉ giữ lại những trang có nội dung sau khi dọn dẹp
        if len(processed_text.strip()) > 20: 
            new_doc = Document(
                page_content=processed_text,
                metadata=doc.metadata.copy() # Tránh tham chiếu mutable ngầm
            )
            
            new_doc.metadata["parsed"] = True
            parsed_docs.append(new_doc)
            
    logger.info(f"[PARSER] Parsing hoàn tất. Giữ lại {len(parsed_docs)} trang hợp lệ.")
    return parsed_docs

if __name__ == "__main__":
    # Test thử nhanh logic parser với các case thực tế
    test_text = """Espressif Systems Confidential V1.2
    
    5.1 Kết nối MQTT
    Bản tin CONNECT dùng để thiết lập kết nối giữa client và broker.
    Thuật ngữ esp_err_t thường được dùng để trả về mã lỗi trong ESP-IDF.
    
    #include <esp_wifi.h>
    void app_main() {
        printf("Hello World\n");
    }
    
    Page 42"""
    
    doc = Document(page_content=test_text, metadata={"source": "test.pdf"})
    result = parse_documents([doc])
    
    print("Nội dung sau khi Parse:\n", result[0].page_content)