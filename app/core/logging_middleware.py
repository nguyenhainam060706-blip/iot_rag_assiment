import json
import logging
import time
import uuid
from pathlib import Path
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from app.config import settings
# --- 1. Setup thư mục & logger dùng chung với exceptions.py ---
Path(settings.LOG_DIR).mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("iot_assistant")
logger.setLevel(settings.LOG_LEVEL)
if not logger.handlers:
    file_handler = logging.FileHandler(Path(settings.LOG_DIR) / "app.log", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()
        session_id = request.headers.get("x-session-id", "unknown")

        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            total_time = round(time.perf_counter() - start, 3)
            status_code = response.status_code if response else 500

            log_entry = {
                "request_id": request_id,
                "session_id": session_id,
                "path": request.url.path,
                "method": request.method,
                "status_code": status_code,
                "total_time": total_time,
            }
            logger.info(json.dumps(log_entry, ensure_ascii=False))