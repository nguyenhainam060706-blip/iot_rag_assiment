from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# Một Dictionary (từ điển) lưu trữ lịch sử trên RAM.
# Ở bản MVP chạy local, dùng RAM là đủ nhanh và tối ưu. 
# Cấu trúc: { "student_001": <lịch_sử_chat>, "student_002": <lịch_sử_chat> }
session_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Hàm này được gọi mỗi khi có request gửi lên API.
    Nó kiểm tra xem sinh viên (session_id) này đã chat bao giờ chưa.
    Nếu chưa, tạo một bộ nhớ mới tinh. Nếu rồi, lấy bộ nhớ cũ ra.
    """
    if session_id not in session_store:
        print(f"[Memory] Tạo phiên hội thoại mới cho: {session_id}")
        session_store[session_id] = InMemoryChatMessageHistory()
    else:
        print(f"[Memory] Tải lại lịch sử hội thoại cho: {session_id}")
        
    return session_store[session_id]

def clear_session_history(session_id: str) -> bool:
    """
    (Tùy chọn) Xóa bộ nhớ của một sinh viên khi họ kết thúc bài Lab,
    giúp giải phóng RAM cho hệ thống.
    """
    if session_id in session_store:
        del session_store[session_id]
        print(f"[Memory] Đã xóa lịch sử của: {session_id}")
        return True
    return False

def get_all_active_sessions() -> list:
    """
    (Tùy chọn) Hàm hỗ trợ để xem hiện tại có bao nhiêu sinh viên đang dùng bot.
    """
    return list(session_store.keys())