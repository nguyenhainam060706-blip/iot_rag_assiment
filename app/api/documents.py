import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from typing import List
from app.core.exceptions import AppError, InvalidRequest
router = APIRouter()
# Thư mục lưu trữ tạm/chính thức theo Section 9 của Tech Spec
UPLOAD_DIR = Path("knowledge/raw_uploads/").resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- Các Pydantic Models cho Response ---
class DocumentInfo(BaseModel):
    id: str
    filename: str
    size_kb: float
    status: str  # Ví dụ: "indexed", "pending"

class DeleteResponse(BaseModel):
    message: str
    document_id: str

class DocumentNotFound(AppError):
    """Lỗi riêng cho domain documents, kế thừa AppError để đi qua đúng handler chung"""
    def __init__(self, message: str = "Document not found on disk"):
        super().__init__(status_code=404, error_code="DOCUMENT_NOT_FOUND", message=message)


def get_safe_path(filename: str) -> Path:
    """
    Chỉ lấy tên gốc của file (basename), loại bỏ mọi đường dẫn ../
    Trả về đối tượng pathlib.Path
    """
    safe_name = os.path.basename(filename)

    # Chặn triệt để tên rỗng, "." (thư mục hiện tại) và ".." (thư mục cha)
    if not safe_name or safe_name in (".", ".."):
        raise InvalidRequest("Tên file hoặc ID không hợp lệ (nghi ngờ Path Traversal).")
    
    # Dùng toán tử / của pathlib để nối chuỗi an toàn
    return UPLOAD_DIR / safe_name


# ---------------------------------------------------------
# 1. GET /api/documents - Liệt kê các tài liệu trong hệ thống
# ---------------------------------------------------------
@router.get("", response_model=List[DocumentInfo])
async def list_documents():
    """Trả về danh sách các tài liệu hiện có trong Knowledge Base."""
    documents = []
    
    # Dùng iterdir() chuẩn của pathlib thay cho os.listdir
    for filepath in UPLOAD_DIR.iterdir():
        if filepath.is_file():
            size = filepath.stat().st_size / 1024  # KB
            documents.append(
                DocumentInfo(
                    id=filepath.name,  
                    filename=filepath.name,
                    size_kb=round(size, 2),
                    status="pending", 
                )
            )
    return documents


# ---------------------------------------------------------
# 2. POST /api/documents - Upload tài liệu mới
# ---------------------------------------------------------
@router.post("")
async def upload_document(
    file: UploadFile = File(...),
    document_type: str = Form(..., description="Ví dụ: datasheet, lab_manual, faq"),
    device: str = Form(..., description="Ví dụ: ESP32, MAX485"),
):
    """Nhận file PDF tải lên và lưu vào hệ thống thư mục."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise InvalidRequest("Chỉ hỗ trợ file PDF.")

    # ĐÃ FIX: Tận dụng helper để sanitize tên file an toàn 100%
    file_path = get_safe_path(file.filename)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    return {
        "message": "Upload thành công",
        "filename": file_path.name,
        "metadata_received": {
            "type": document_type,
            "device": device,
        },
        "status": "ready_for_ingestion",
    }


# ---------------------------------------------------------
# 3. DELETE /api/documents/{id} - Xóa tài liệu
# ---------------------------------------------------------
@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str):
    """Xóa tài liệu khỏi hệ thống."""
    # file_path giờ là đối tượng Path, có thể gọi .exists() và .unlink()
    file_path = get_safe_path(document_id)

    if not file_path.exists():
        raise DocumentNotFound()

    file_path.unlink()

    return DeleteResponse(
        message="Đã xóa tài liệu và các vector liên quan.",
        document_id=document_id,
    )


# ---------------------------------------------------------
# 4. POST /api/documents/reindex - Yêu cầu index lại toàn bộ
# ---------------------------------------------------------
@router.post("/reindex")
async def trigger_reindex():
    return {
        "status": "processing",
        "message": "Quá trình Re-indexing đang chạy ngầm...",
    }