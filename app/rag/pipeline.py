import concurrent.futures
import threading
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
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
print(f"Loading {settings.EMBEDDING_MODEL} Embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name=settings.EMBEDDING_MODEL
)

print("Connecting to ChromaDB...")
vector_db = Chroma(
    persist_directory=settings.CHROMA_PERSIST_DIR,
    embedding_function=embeddings,
    collection_name=settings.CHROMA_COLLECTION,
)

# LƯU Ý: bỏ vector_db.as_retriever() vì retriever mặc định của LangChain
# không trả kèm similarity score trong metadata -> field "score" trong
# response API sẽ luôn là 0.0, sai với yêu cầu mục 27/36 của spec.
# Thay vào đó, ta gọi trực tiếp vector_db.similarity_search_with_score()
# ở bước retrieval bên dưới để có score thật.


# ==========================================
# 2. LOCAL LLM - OLLAMA
# ==========================================
print(f"Connecting to Local LLM ({settings.LLM_MODEL})...")
llm = ChatOllama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.LLM_MODEL,
    temperature=settings.LLM_TEMPERATURE,
)


# ==========================================
# 3. BỘ NHỚ RAM TẠM THỜI (THREAD-SAFE)
# ==========================================
session_histories = {}

# run_rag_pipeline chạy trong threadpool của FastAPI (vì được gọi bằng
# 'def' thường trong chat.py) -> nhiều request có thể đọc/ghi cùng lúc
# vào session_histories. Dùng Lock để tránh race condition khi 2 sinh
# viên cùng session_id gửi câu hỏi gần như đồng thời.
_history_lock = threading.Lock()

def get_chat_history(session_id: str):
    """Lấy lịch sử hội thoại của một session. Nếu chưa có thì tạo mới."""
    with _history_lock:
        if session_id not in session_histories:
            session_histories[session_id] = []
        # Trả về bản copy để tránh code bên ngoài lock vô tình sửa list gốc
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

# Chain viết lại câu hỏi: prompt -> LLM -> chuỗi text thuần.
# Tách riêng ra thay vì dùng create_history_aware_retriever để có thể
# tự kiểm soát bước retrieval phía sau và lấy được similarity score.
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

# create_stuff_documents_chain trả về CHUỖI TEXT thuần (có StrOutputParser
# sẵn bên trong), không phải dict như create_retrieval_chain trả ra trước đây.
qa_chain = create_stuff_documents_chain(llm, qa_prompt)


# ==========================================
# 6. HELPER: GỌI LLM CÓ TIMEOUT + PHÂN LOẠI LỖI
# ==========================================
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

def _invoke_llm_chain(chain, payload: dict, timeout_seconds: float):
    """
    Chạy 1 lời gọi LLM (rewrite hoặc generate) với timeout, và phân loại
    đúng lỗi theo mục 30:
      - Quá thời gian  -> RequestTimeout (504)
      - Lỗi kết nối/bug khác từ Ollama -> LLMServiceUnavailable (503)
    Không để lỗi thô (ConnectionError, httpx.ConnectError...) rơi thẳng
    vào global_exception_handler chung chung.
    """
    future = _executor.submit(chain.invoke, payload)
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        raise RequestTimeout(
            f"LLM ({settings.LLM_MODEL}) không phản hồi sau {timeout_seconds}s."
        )
    except Exception as e:
        raise LLMServiceUnavailable(f"Lỗi khi gọi Ollama: {str(e)}")


def _retrieve_documents_with_score(query: str, k: int):
    """
    Truy vấn ChromaDB, trả về list Document đã gắn kèm score thật vào
    metadata. Mọi lỗi kết nối/đọc ChromaDB -> VectorDBUnavailable (503),
    tách biệt rõ ràng với lỗi từ phía LLM.
    """
    try:
        results = vector_db.similarity_search_with_score(query, k=k)
    except Exception as e:
        raise VectorDBUnavailable(f"Lỗi khi truy vấn ChromaDB: {str(e)}")
    
    documents = []
    for doc, score in results:
        # score trả về từ Chroma là "distance" (càng thấp càng giống);
        # gắn thẳng vào metadata để tầng build sources phía dưới dùng lại.
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
    """
    Hàm chính được chat.py gọi (POST /api/chat).
    Input:
        session_id, question, code, log
    Output:
        (answer: str, sources: list[dict]) -- TUPLE, đúng như chat.py
        đang unpack: `answer, raw_sources = run_rag_pipeline(...)`
    """
    code_text = code if code else "Không có."
    log_text = log if log else "Không có."
    current_history = get_chat_history(session_id)
    
    # --------------------------------------
    # BƯỚC 1: Viết lại câu hỏi cho rõ ngữ cảnh
    # (chỉ cần khi đã có lịch sử hội thoại, câu hỏi đầu tiên giữ nguyên
    # để tiết kiệm 1 lần gọi LLM không cần thiết)
    # --------------------------------------
    if current_history:
        search_query = _invoke_llm_chain(
            query_rewriter_chain,
            {"input": question, "chat_history": current_history},
            timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
        )
    else:
        search_query = question
        
    # --------------------------------------
    # BƯỚC 2: Truy vấn ChromaDB lấy tài liệu + score thật
    # --------------------------------------
    documents = _retrieve_documents_with_score(
        search_query, k=getattr(settings, "TOP_K_DEFAULT", 5)
    )
    
    # --------------------------------------
    # BƯỚC 3: Sinh câu trả lời từ LLM dựa trên context đã retrieve
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
    # BƯỚC 4: Cập nhật memory hội thoại
    # --------------------------------------
    update_chat_history(session_id, question, final_answer)
    
    # --------------------------------------
    # BƯỚC 5: Build sources theo đúng schema mục 27
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