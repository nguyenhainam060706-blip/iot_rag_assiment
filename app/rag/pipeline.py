from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Old (Broken)
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever

from langchain_core.messages import HumanMessage, AIMessage
from app.config import settings
# ==========================================
# 1. KHỞI TẠO MODELS & VECTOR DB
# ==========================================
print(f"Loading {settings.EMBEDDING_MODEL} Embeddings...")
embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

print("Connecting to ChromaDB...")
vector_db = Chroma(
    persist_directory=settings.CHROMA_PERSIST_DIR, 
    embedding_function=embeddings,
    collection_name=settings.CHROMA_COLLECTION
)
# Lấy 5 tài liệu liên quan nhất
retriever = vector_db.as_retriever(search_kwargs={"k": 5}) 

print(f"Connecting to Local LLM ({settings.LLM_MODEL})...")
llm = ChatOllama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.LLM_MODEL, 
    temperature=settings.LLM_TEMPERATURE
)

# ==========================================
# 2. BỘ NHỚ RAM TẠM THỜI (Tối ưu VRAM)
# ==========================================
session_histories = {}

def get_chat_history(session_id: str):
    if session_id not in session_histories:
        session_histories[session_id] = []
    return session_histories[session_id]

def update_chat_history(session_id: str, question: str, answer: str):
    history = get_chat_history(session_id)
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))
    
    # Cắt bộ nhớ giữ lại tối đa 3 lượt hội thoại (6 messages) để chống tràn VRAM
    if len(history) > 6:
        session_histories[session_id] = history[-6:]

# ==========================================
# 3. XÂY DỰNG HISTORY-AWARE RAG CHAIN
# ==========================================

# 3.1. Prompt báo cho LLM viết lại câu hỏi tìm kiếm
contextualize_q_system_prompt = (
    "Dựa trên lịch sử trò chuyện và câu hỏi mới nhất, hãy viết lại câu hỏi "
    "thành một câu độc lập, đầy đủ từ khóa để tìm kiếm trong tài liệu.\n"
    "KHÔNG trả lời câu hỏi, chỉ trả về câu hỏi đã được viết lại."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Bộ tìm kiếm thông minh: Tự đọc history -> Viết lại câu hỏi -> Tìm ChromaDB
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# 3.2. Prompt chính để trả lời (Có hỗ trợ phân tích code/log theo Section 23, 24)
qa_system_prompt = """Bạn là 'AI Teaching Assistant for IoT Practical Laboratory'.
Nhiệm vụ của bạn là hỗ trợ sinh viên thực hành IoT dựa trên tài liệu Lab.

QUY TẮC NGHIÊM NGẶT:
1. KHÔNG tự tạo thông số kỹ thuật, pinout, voltage, hoặc bịa datasheet.
2. NẾU TÀI LIỆU KHÔNG CÓ, HÃY NÓI CHÍNH XÁC: 'Không đủ thông tin trong Knowledge Base'.
3. Hướng dẫn từng bước, giải thích nguyên nhân trước khi đưa giải pháp.
4. Trả lời bằng Tiếng Việt thân thiện và có thể tìm hiểu thêm từ tài liệu Tiếng Anh rồi đưa về Tiếng Việt thân thiện.


TÀI LIỆU KNOWLEDGE BASE:
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    # Gộp chung câu hỏi, code và log vào input của Human
    ("human", "Câu hỏi: {input}\n\nCode đang chạy:\n{code}\n\nLog lỗi hệ thống:\n{log}"),
])

# Chuỗi trả lời kết hợp tài liệu
qa_chain = create_stuff_documents_chain(llm, qa_prompt)

# Ghép toàn bộ luồng lại
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# ==========================================
# 4. HÀM CHÍNH API GỌI VÀO
# ==========================================
def ask_iot_assistant(session_id: str, question: str, code: str = None, log: str = None):
    # Xử lý input rỗng
    code_text = code if code else "Không có."
    log_text = log if log else "Không có."
    
    current_history = get_chat_history(session_id)
    
    # Kích hoạt Chain
    response = rag_chain.invoke({
        "input": question,
        "chat_history": current_history, 
        "code": code_text,
        "log": log_text
    })
    
    final_answer = response["answer"]
    
    # Cập nhật lịch sử (đã tự động có giới hạn 6 tin nhắn)
    update_chat_history(session_id, question, final_answer)
    
    # Trích xuất nguồn (Source Citation)
    sources = []
    for doc in response["context"]:
        metadata = doc.metadata
        sources.append({
            "document": metadata.get("source", "Unknown Document"),
            "page": metadata.get("page", 0),
            "score": metadata.get("score", 0.0) # Có thể Chroma chưa cấp score, cứ để mặc định 0.0
        })
        
    return {
        "answer": final_answer,
        "sources": sources,
        "session_id": session_id
    }