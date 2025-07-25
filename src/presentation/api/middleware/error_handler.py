import time
import traceback
from typing import Callable
from uuid import uuid4

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


async def error_handler_middleware(request: Request, call_next: Callable) -> Response:
    """Global error handler middleware."""
    request_id = str(uuid4())
    request.state.request_id = request_id
    
    try:
        response = await call_next(request)
        return response
    
    except ValueError as e:
        # Handle validation errors
        return JSONResponse(
            status_code=400,
            content={
                "error": "Bad Request",
                "message": str(e),
                "request_id": request_id,
                "type": "validation_error"
            }
        )
    
    except PermissionError as e:
        # Handle permission errors
        return JSONResponse(
            status_code=403,
            content={
                "error": "Forbidden",
                "message": str(e),
                "request_id": request_id,
                "type": "permission_error"
            }
        )
    
    except FileNotFoundError as e:
        # Handle not found errors
        return JSONResponse(
            status_code=404,
            content={
                "error": "Not Found",
                "message": str(e),
                "request_id": request_id,
                "type": "not_found_error"
            }
        )
    
    except TimeoutError as e:
        # Handle timeout errors
        return JSONResponse(
            status_code=504,
            content={
                "error": "Gateway Timeout",
                "message": "The request took too long to process",
                "request_id": request_id,
                "type": "timeout_error"
            }
        )
    
    except Exception as e:
        # Handle unexpected errors
        print(f"Unexpected error in request {request_id}: {traceback.format_exc()}")
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
                "request_id": request_id,
                "type": "internal_error"
            }
        )