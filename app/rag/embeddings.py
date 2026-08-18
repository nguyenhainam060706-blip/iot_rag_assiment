from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    # Sử dụng BGE-M3: Hỗ trợ đa ngôn ngữ, tiếng Việt cực tốt cho tài liệu kỹ thuật
    print("Loading BGE-M3 Embeddings...")
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'} # Đổi thành 'cuda' nếu đã cài chuẩn driver Nvidia cho GTX 1650
    )