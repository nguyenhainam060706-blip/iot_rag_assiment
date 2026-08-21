from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import HumanMessage, AIMessage

from langchain.chains import (
    create_retrieval_chain,
    create_history_aware_retriever,
)
from langchain.chains.combine_documents import create_stuff_documents_chain

from app.config import settings


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

# Lấy 5 tài liệu liên quan nhất
retriever = vector_db.as_retriever(
    search_kwargs={"k": 5}
)


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
# 3. BỘ NHỚ RAM TẠM THỜI
# ==========================================

session_histories = {}


def get_chat_history(session_id: str):
    """
    Lấy lịch sử hội thoại của một session.
    Nếu session chưa tồn tại thì tạo mới.
    """
    if session_id not in session_histories:
        session_histories[session_id] = []

    return session_histories[session_id]


def update_chat_history(
    session_id: str,
    question: str,
    answer: str,
):
    """
    Lưu câu hỏi + câu trả lời vào RAM.

    Giữ tối đa 3 lượt hội thoại
    = 6 messages.
    """

    history = get_chat_history(session_id)

    history.append(
        HumanMessage(content=question)
    )

    history.append(
        AIMessage(content=answer)
    )

    # Giữ lại tối đa 6 messages
    if len(history) > 6:
        session_histories[session_id] = history[-6:]


# ==========================================
# 4. HISTORY-AWARE RETRIEVER
# ==========================================

# Prompt dùng để viết lại câu hỏi
# dựa trên lịch sử hội thoại.

contextualize_q_system_prompt = """
Dựa trên lịch sử trò chuyện và câu hỏi mới nhất,
hãy viết lại câu hỏi thành một câu độc lập,
đầy đủ từ khóa để tìm kiếm trong tài liệu.

Nếu câu hỏi đã đầy đủ ngữ cảnh thì giữ nguyên ý nghĩa.

KHÔNG trả lời câu hỏi.
CHỈ trả về câu hỏi tìm kiếm đã được viết lại.
"""


contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            contextualize_q_system_prompt,
        ),

        MessagesPlaceholder(
            variable_name="chat_history"
        ),

        (
            "human",
            "{input}",
        ),
    ]
)


# History
#    ↓
# Rewrite question
#    ↓
# ChromaDB
#    ↓
# Relevant documents

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt,
)


# ==========================================
# 5. PROMPT TRẢ LỜI
# ==========================================

qa_system_prompt = """
Bạn là "AI Teaching Assistant for IoT Practical Laboratory".

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

{context}
"""


qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            qa_system_prompt,
        ),

        MessagesPlaceholder(
            variable_name="chat_history"
        ),

        (
            "human",
            """
Câu hỏi:
{input}

Code đang chạy:
{code}

Log lỗi hệ thống:
{log}
""",
        ),
    ]
)


# ==========================================
# 6. DOCUMENT QA CHAIN
# ==========================================

qa_chain = create_stuff_documents_chain(
    llm,
    qa_prompt,
)


# ==========================================
# 7. RAG CHAIN
# ==========================================

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    qa_chain,
)


# ==========================================
# 8. HÀM API CHÍNH
# ==========================================

def ask_iot_assistant(
    session_id: str,
    question: str,
    code: str = None,
    log: str = None,
):
    """
    Hàm chính được API gọi để hỏi AI.

    Input:
        session_id : ID của cuộc hội thoại
        question   : câu hỏi
        code       : code người dùng cung cấp
        log        : log lỗi người dùng cung cấp

    Output:
        answer
        sources
        session_id
    """

    # --------------------------------------
    # Xử lý input rỗng
    # --------------------------------------

    code_text = code if code else "Không có."

    log_text = log if log else "Không có."


    # --------------------------------------
    # Lấy lịch sử hội thoại
    # --------------------------------------

    current_history = get_chat_history(
        session_id
    )


    # --------------------------------------
    # Chạy RAG
    # --------------------------------------

    response = rag_chain.invoke(
        {
            "input": question,

            "chat_history": current_history,

            "code": code_text,

            "log": log_text,
        }
    )


    # --------------------------------------
    # Lấy câu trả lời
    # --------------------------------------

    final_answer = response["answer"]


    # --------------------------------------
    # Cập nhật memory
    # --------------------------------------

    update_chat_history(
        session_id,
        question,
        final_answer,
    )


    # --------------------------------------
    # Lấy sources
    # --------------------------------------

    sources = []

    for doc in response.get("context", []):

        metadata = doc.metadata

        sources.append(
            {
                "document": metadata.get(
                    "source",
                    "Unknown Document",
                ),

                "page": metadata.get(
                    "page",
                    0,
                ),

                "score": metadata.get(
                    "score",
                    0.0,
                ),
            }
        )


    # --------------------------------------
    # Return API response
    # --------------------------------------

    return {
        "answer": final_answer,

        "sources": sources,

        "session_id": session_id,
    }