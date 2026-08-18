from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

# ==========================================
# 1. KHỞI TẠO MODELS
# ==========================================
print("Loading BGE-M3 Embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

print("Connecting to ChromaDB...")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

print("Connecting to Local LLM (Qwen)...")
llm = ChatOllama(model="qwen:4b", temperature=0)

# ==========================================
# 2. BỘ NHỚ RAM TẠM THỜI (MEMORY MANAGER)
# ==========================================
# Đây là cuốn sổ tay lưu lịch sử chat của toàn bộ sinh viên
session_histories = {}

def get_chat_history(session_id: str):
    """Lấy lịch sử chat của một sinh viên cụ thể"""
    if session_id not in session_histories:
        session_histories[session_id] = []
    return session_histories[session_id]

def update_chat_history(session_id: str, question: str, answer: str):
    """Lưu lại câu hỏi và câu trả lời vào sổ tay"""
    history = get_chat_history(session_id)
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))
    
    # Mẹo tối ưu VRAM: Chỉ nhớ 3 cặp câu hỏi-trả lời gần nhất (6 messages)
    if len(history) > 6:
        session_histories[session_id] = history[-6:]

# ==========================================
# 3. XÂY DỰNG PROMPT (CÓ CHỖ CHO LỊCH SỬ)
# ==========================================
system_prompt = """Bạn là 'AI Teaching Assistant for IoT Practical Laboratory'.
Nhiệm vụ của bạn là hỗ trợ sinh viên thực hành IoT dựa trên tài liệu Lab.

QUY TẮC NGHIÊM NGẶT:
1. KHÔNG tự tạo thông số kỹ thuật, pinout, voltage, hoặc bịa datasheet.
2. NẾU TÀI LIỆU KHÔNG CÓ, HÃY NÓI: 'Không đủ thông tin trong Knowledge Base'.
3. Hướng dẫn từng bước, giải thích nguyên nhân trước khi đưa giải pháp.
4. Trả lời bằng Tiếng Việt thân thiện.

TÀI LIỆU KNOWLEDGE BASE:
{context}

THÔNG TIN TỪ MÁY SINH VIÊN (Nếu có):
Code: {code}
Log lỗi: {log}
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    # Dòng này cực kỳ quan trọng: Chèn lịch sử chat vào trước câu hỏi mới
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

rag_chain = create_retrieval_chain(
    retriever, 
    create_stuff_documents_chain(llm, prompt_template)
)

# ==========================================
# 4. HÀM CHÍNH API GỌI VÀO
# ==========================================
def ask_iot_assistant(session_id: str, question: str, code: str = None, log: str = None):
    code_text = code if code else "Không có code được gửi kèm."
    log_text = log if log else "Không có log lỗi."
    
    # 4.1. Lấy lịch sử chat cũ của sinh viên này
    current_history = get_chat_history(session_id)
    
    # 4.2. Chạy chuỗi RAG (Nhồi thêm chat_history vào)
    response = rag_chain.invoke({
        "input": question,
        "chat_history": current_history, # AI sẽ đọc được các câu trước đó
        "code": code_text,
        "log": log_text
    })
    
    # 4.3. Lưu ngay kết quả vừa trả lời vào bộ nhớ để lần sau dùng tiếp
    final_answer = response["answer"]
    update_chat_history(session_id, question, final_answer)
    
    # 4.4. Trích xuất nguồn tài liệu
    sources = []
    for doc in response["context"]:
        metadata = doc.metadata
        sources.append({
            "document": metadata.get("source", "Unknown Document"),
            "page": metadata.get("page", 0),
            "score": 0.0 
        })
        
    return {
        "answer": final_answer,
        "sources": sources,
        "session_id": session_id
    }