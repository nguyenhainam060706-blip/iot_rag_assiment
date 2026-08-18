from langchain_ollama import ChatOllama

def get_llm():
    print("Connecting to Local LLM (Qwen)...")
    return ChatOllama(
        model="qwen2.5:3b", # Chạy bản 3B hoặc 4B để tối ưu cho 4GB VRAM
        temperature=0,      # BẮT BUỘC = 0 để AI không bịa đặt thông số (Section 21)
        max_tokens=1024
    )