import concurrent.futures
import threading
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
# BỎ DÒNG NÀY: from langchain_huggingface import HuggingFaceEmbeddings
# THÊM 2 DÒNG NÀY: Gọi Singleton và Logger
from app.rag.embeddings import get_embeddings
from app.core.logging_middleware import logger

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain

from app.config import settings
from app.core.exceptions import (
    LLMServiceUnavailable,
    VectorDBUnavailable,
    RequestTimeout,
)

# ==========================================
# 1. KHỞI TẠO MODELS & VECTOR DB
# ==========================================
logger.info(f"[PIPELINE] Lấy instance BGE-M3 Embeddings (Singleton)...")
# CHỈNH SỬA QUAN TRỌNG NHẤT NẰM Ở ĐÂY: Sử dụng Singleton để chia sẻ chung VRAM
embeddings = get_embeddings()

logger.info("[PIPELINE] Connecting to ChromaDB...")
vector_db = Chroma(
    persist_directory=settings.CHROMA_PERSIST_DIR,
    embedding_function=embeddings,
    collection_name=settings.CHROMA_COLLECTION,
)

# ==========================================
# 2. LOCAL LLM - OLLAMA
# ==========================================
logger.info(f"[PIPELINE] Connecting to Local LLM ({settings.LLM_MODEL})...")
llm = ChatOllama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.LLM_MODEL,
    temperature=settings.LLM_TEMPERATURE,
)

# ==========================================
# 3. BỘ NHỚ RAM TẠM THỜI (THREAD-SAFE)
# ==========================================
session_histories = {}
_history_lock = threading.Lock()

def get_chat_history(session_id: str):
    """Lấy lịch sử hội thoại của một session. Nếu chưa có thì tạo mới."""
    with _history_lock:
        if session_id not in session_histories:
            session_histories[session_id] = []
        return list(session_histories[session_id])

def update_chat_history(session_id: str, question: str, answer: str):
    """Lưu câu hỏi + câu trả lời vào RAM. Giữ tối đa 3 lượt hội thoại (6 messages)."""
    with _history_lock:
        history = session_histories.setdefault(session_id, [])
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))
        if len(history) > 6:
            session_histories[session_id] = history[-6:]

# ==========================================
# 4. PROMPT VIẾT LẠI CÂU HỎI (CONTEXTUALIZE)
# ==========================================
contextualize_q_system_prompt = """Dựa trên lịch sử trò chuyện và câu hỏi mới nhất,
hãy viết lại câu hỏi thành một câu độc lập,
đầy đủ từ khóa để tìm kiếm trong tài liệu.
Nếu câu hỏi đã đầy đủ ngữ cảnh thì giữ nguyên ý nghĩa.
KHÔNG trả lời câu hỏi.
CHỈ trả về câu hỏi tìm kiếm đã được viết lại."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

query_rewriter_chain = contextualize_q_prompt | llm | StrOutputParser()

# ==========================================
# 5. PROMPT TRẢ LỜI
# ==========================================
qa_system_prompt = """Bạn là "AI Teaching Assistant for IoT Practical Laboratory".
Nhiệm vụ của bạn là hỗ trợ sinh viên thực hành IoT
dựa trên Knowledge Base của phòng Lab.
QUY TẮC NGHIÊM NGẶT:
1. KHÔNG tự tạo thông số kỹ thuật,
   pinout, voltage, current hoặc datasheet.
2. Nếu Knowledge Base không có thông tin cần thiết,
   hãy nói chính xác:
   "Không đủ thông tin trong Knowledge Base"
3. Khi hướng dẫn kỹ thuật:
   - Giải thích nguyên nhân trước.
   - Sau đó đưa ra giải pháp.
   - Hướng dẫn từng bước.
4. Nếu người dùng cung cấp code:
   - Phân tích code.
   - Chỉ ra lỗi.
   - Giải thích tại sao lỗi xảy ra.
   - Đưa code sửa nếu cần.
5. Nếu người dùng cung cấp log:
   - Phân tích log.
   - Xác định lỗi chính.
   - Giải thích nguyên nhân.
   - Đưa hướng xử lý.
6. Không được coi code hoặc log của người dùng
   là Knowledge Base.
7. Trả lời bằng Tiếng Việt thân thiện,
   dễ hiểu đối với sinh viên.
8. Có thể sử dụng tài liệu tiếng Anh trong Knowledge Base,
   nhưng phải giải thích lại bằng Tiếng Việt.
9. Chỉ sử dụng thông tin có trong Knowledge Base
   để khẳng định các thông tin kỹ thuật quan trọng.

TÀI LIỆU KNOWLEDGE BASE:
{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            """Câu hỏi:
{input}

Code đang chạy:
{code}

Log lỗi hệ thống:
{log}""",
        ),
    ]
)

qa_chain = create_stuff_documents_chain(llm, qa_prompt)

# ==========================================
# 6. HELPER: GỌI LLM CÓ TIMEOUT + PHÂN LOẠI LỖI
# ==========================================
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

def _invoke_llm_chain(chain, payload: dict, timeout_seconds: float):
    future = _executor.submit(chain.invoke, payload)
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        logger.error(f"[PIPELINE] LLM Timeout sau {timeout_seconds}s.")
        raise RequestTimeout(
            f"LLM ({settings.LLM_MODEL}) không phản hồi sau {timeout_seconds}s."
        )
    except Exception as e:
        logger.error(f"[PIPELINE] LLM Error: {str(e)}", exc_info=True)
        raise LLMServiceUnavailable(f"Lỗi khi gọi Ollama: {str(e)}")

def _retrieve_documents_with_score(query: str, k: int):
    try:
        results = vector_db.similarity_search_with_score(query, k=k)
    except Exception as e:
        logger.error(f"[PIPELINE] ChromaDB Query Error: {str(e)}", exc_info=True)
        raise VectorDBUnavailable(f"Lỗi khi truy vấn ChromaDB: {str(e)}")
    
    documents = []
    for doc, score in results:
        doc.metadata = {**doc.metadata, "score": float(score)}
        documents.append(doc)
    return documents

# ==========================================
# 7. HÀM API CHÍNH
# ==========================================
def run_rag_pipeline(
    session_id: str,
    question: str,
    code: str = None,
    log: str = None,
):
    code_text = code if code else "Không có."
    log_text = log if log else "Không có."
    current_history = get_chat_history(session_id)
    
    # --------------------------------------
    # BƯỚC 1: Viết lại câu hỏi
    # --------------------------------------
    if current_history:
        search_query = _invoke_llm_chain(
            query_rewriter_chain,
            {"input": question, "chat_history": current_history},
            timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
        )
        logger.info(f"[PIPELINE] Câu hỏi đã viết lại: '{search_query}'")
    else:
        search_query = question
        
    # --------------------------------------
    # BƯỚC 2: Truy vấn ChromaDB lấy tài liệu + score
    # --------------------------------------
    documents = _retrieve_documents_with_score(
        search_query, k=getattr(settings, "TOP_K_DEFAULT", 5)
    )
    
    # --------------------------------------
    # BƯỚC 3: Sinh câu trả lời từ LLM
    # --------------------------------------
    final_answer = _invoke_llm_chain(
        qa_chain,
        {
            "input": question,
            "chat_history": current_history,
            "code": code_text,
            "log": log_text,
            "context": documents,
        },
        timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
    )
    
    # --------------------------------------
    # BƯỚC 4: Cập nhật memory
    # --------------------------------------
    update_chat_history(session_id, question, final_answer)
    
    # --------------------------------------
    # BƯỚC 5: Build sources
    # --------------------------------------
    sources = []
    for doc in documents:
        metadata = doc.metadata
        sources.append(
            {
                "document": metadata.get("source", "Unknown Document"),
                "page": metadata.get("page", 0),
                "score": metadata.get("score", 0.0),
            }
        )
        
    return final_answer, sources