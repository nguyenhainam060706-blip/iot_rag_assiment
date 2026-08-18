from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_index():
    # Sử dụng đúng Embedding model BGE-M3 theo Section 5.3
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    
    # Dùng PyMuPDFLoader theo yêu cầu Section 11.1
    loader = PyMuPDFLoader("knowledge/kit/esp32/ESP32_Datasheet.pdf")
    docs = loader.load()

    # Chunking Strategy theo Section 12 (500-1000 tokens)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(docs)

    # Thêm Metadata Schema theo Section 13
    for chunk in chunks:
        chunk.metadata["source"] = "ESP32_Datasheet.pdf"
        chunk.metadata["document_type"] = "datasheet"
        chunk.metadata["device"] = "ESP32"

    # Lưu vào ChromaDB theo Section 15
    Chroma.from_documents(
        chunks, embeddings, persist_directory="./chroma_db", collection_name="iot_knowledge"
    )
    print("Ingestion Pipeline Hoàn Tất!")