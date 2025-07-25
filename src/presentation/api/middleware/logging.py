import time
import json
from typing import Callable

from fastapi import Request, Response


async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """Log requests and responses."""
    start_time = time.time()
    
    # Get request ID from state or generate new one
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Log request
    request_log = {
        "timestamp": time.time(),
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "query_params": dict(request.query_params),
        "client": request.client.host if request.client else "unknown"
    }
    
    print(f"📥 Request: {json.dumps(request_log)}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Add custom headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log response
    response_log = {
        "timestamp": time.time(),
        "request_id": request_id,
        "status_code": response.status_code,
        "process_time": process_time,
        "path": request.url.path
    }
    
    print(f"📤 Response: {json.dumps(response_log)}")
    
    return response