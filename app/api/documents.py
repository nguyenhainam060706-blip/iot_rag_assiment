import os
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List

router = APIRouter()

# Thư mục lưu trữ tạm/chính thức theo Section 9 của Tech Spec
UPLOAD_DIR = "knowledge/raw_uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Các Pydantic Models cho Response ---
class DocumentInfo(BaseModel):
    id: str
    filename: str
    size_kb: float
    status: str # Ví dụ: "indexed", "pending"

class DeleteResponse(BaseModel):
    message: str
    document_id: str

# ---------------------------------------------------------
# 1. GET /api/documents - Liệt kê các tài liệu trong hệ thống
# ---------------------------------------------------------
@router.get("/", response_model=List[DocumentInfo])
async def list_documents():
    """
    Trả về danh sách các tài liệu hiện có trong Knowledge Base.
    (Trong thực tế, bạn sẽ query database ChromaDB hoặc SQLite để lấy danh sách này)
    """
    try:
        documents = []
        # Khung ví dụ đọc từ thư mục upload
        for filename in os.listdir(UPLOAD_DIR):
            filepath = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath) / 1024 # KB
                documents.append(
                    DocumentInfo(
                        id=filename, # Tạm dùng tên file làm ID cho MVP
                        filename=filename,
                        size_kb=round(size, 2),
                        status="pending" # Trạng thái chờ pipeline Ingestion xử lý
                    )
                )
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi đọc danh sách tài liệu: {str(e)}")


# ---------------------------------------------------------
# 2. POST /api/documents - Upload tài liệu mới
# ---------------------------------------------------------
@router.post("/")
async def upload_document(
    file: UploadFile = File(...),
    document_type: str = Form(..., description="Ví dụ: datasheet, lab_manual, faq"),
    device: str = Form(..., description="Ví dụ: ESP32, MAX485")
):
    """
    Nhận file PDF tải lên và lưu vào hệ thống thư mục.
    Sau khi tải lên thành công, hệ thống nên trigger (gọi) pipeline Ingestion (Section 29).
    """
    if not file.filename.endswith('.pdf'):
         raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file PDF.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        # Lưu file xuống đĩa
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # TƯƠNG LAI: Gọi background task chạy Ingestion Pipeline ở đây
        # ví dụ: background_tasks.add_task(process_pdf_and_index, file_path, document_type, device)

        return {
            "message": "Upload thành công",
            "filename": file.filename,
            "metadata_received": {
                "type": document_type,
                "device": device
            },
            "status": "ready_for_ingestion"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lưu file: {str(e)}")


# ---------------------------------------------------------
# 3. DELETE /api/documents/{id} - Xóa tài liệu
# ---------------------------------------------------------
@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str):
    """
    Xóa tài liệu khỏi hệ thống.
    Cần xóa cả file vật lý lẫn các chunk tương ứng trong ChromaDB.
    """
    file_path = os.path.join(UPLOAD_DIR, document_id) # document_id tạm là tên file
    
    # 1. Xóa file vật lý
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu trên đĩa.")

    # 2. Xóa khỏi ChromaDB (Cần kết nối với vectorstore)
    # try:
    #     vector_db = Chroma(...)
    #     vector_db.delete(where={"source": document_id})
    # except Exception as e:
    #    ...

    return DeleteResponse(
        message="Đã xóa tài liệu và các vector liên quan.",
        document_id=document_id
    )


# ---------------------------------------------------------
# 4. POST /api/documents/reindex - Yêu cầu index lại toàn bộ
# ---------------------------------------------------------
@router.post("/reindex")
async def trigger_reindex():
    """
    Kích hoạt lại toàn bộ Ingestion Pipeline (xóa index cũ, đọc lại toàn bộ file).
    Chạy khi có sự thay đổi lớn về thư mục tài liệu hoặc thay đổi chunking strategy.
    """
    # TƯƠNG LAI: Gọi logic chạy lại file ingestion_pipeline.py
    
    return {
        "status": "processing",
        "message": "Quá trình Re-indexing đang chạy ngầm..."
    }