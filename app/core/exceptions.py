import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

# 1. Đổi tên logger khớp 100% với logging_middleware.py để dùng chung cấu hình (FileHandler, Formatter...)
logger = logging.getLogger("iot_assistant")

# ==========================================
# 1. ĐỊNH NGHĨA CÁC CLASS LỖI (EXCEPTIONS)
# ==========================================

class AppError(Exception):
    def __init__(self, status_code: int, error_code: str, message: str = ""):
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        super().__init__(self.message)

class LLMServiceUnavailable(AppError):
    def __init__(self, message: str = "Ollama is down or unreachable"):
        super().__init__(status_code=503, error_code="LLM_SERVICE_UNAVAILABLE", message=message)

class VectorDBUnavailable(AppError):
    def __init__(self, message: str = "ChromaDB connection failed"):
        super().__init__(status_code=503, error_code="VECTOR_DATABASE_UNAVAILABLE", message=message)

class InvalidRequest(AppError):
    def __init__(self, message: str = "Question is empty or invalid data"):
        super().__init__(status_code=400, error_code="INVALID_REQUEST", message=message)

class RequestTimeout(AppError):
    def __init__(self, message: str = "Upstream LLM/VectorDB exceeded timeout limit"):
        super().__init__(status_code=504, error_code="REQUEST_TIMEOUT", message=message)


# ==========================================
# 2. ĐĂNG KÝ HANDLER CHO FASTAPI
# ==========================================

def register_exception_handlers(app: FastAPI):
    
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        logger.error(f"[{exc.error_code}] {exc.message} | Path: {request.url.path}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.error_code}
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(f"[VALIDATION_ERROR] {exc.errors()} | Path: {request.url.path}")
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_REQUEST"}
        )

    # 2. ĐÃ FIX: Đè luôn handler mặc định của HTTPException để bảo vệ error contract
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """
        Bắt các lỗi do code lỡ dùng raise HTTPException(...) thay vì raise AppError(...).
        Ép format từ {"detail": "..."} về đúng chuẩn {"error": "..."}
        """
        logger.error(f"[HTTP_EXCEPTION] {exc.detail} | Status: {exc.status_code} | Path: {request.url.path}")
        
        # Xử lý an toàn: nếu detail không phải string (dict/list), ép về string
        error_msg = exc.detail if isinstance(exc.detail, str) else "UNKNOWN_HTTP_ERROR"
        
        return JSONResponse(
            status_code=exc.status_code,
            # Nếu muốn chuẩn hóa chữ HOA_CÓ_GẠCH_DƯỚI, có thể dùng: error_msg.upper().replace(" ", "_")
            content={"error": error_msg} 
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.critical(f"[INTERNAL_SERVER_ERROR] Unhandled exception: {str(exc)} | Path: {request.url.path}")
        return JSONResponse(
            status_code=500,
            content={"error": "INTERNAL_SERVER_ERROR"}
        )