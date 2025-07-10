# Security Best Practices Guide - RAG Comprehensive System

## 🔐 Security Architecture Overview

### Defense in Depth Strategy
```
┌─────────────────────────────────────────────┐
│            WAF / DDoS Protection            │
├─────────────────────────────────────────────┤
│          Load Balancer with SSL             │
├─────────────────────────────────────────────┤
│       API Gateway (Rate Limiting)           │
├─────────────────────────────────────────────┤
│    Authentication & Authorization Layer     │
├─────────────────────────────────────────────┤
│         Application Security                │
├─────────────────────────────────────────────┤
│          Data Encryption Layer              │
├─────────────────────────────────────────────┤
│       Network Security (Firewall)           │
└─────────────────────────────────────────────┘
```

### Security Principles
1. **Zero Trust**: Never trust, always verify
2. **Least Privilege**: Minimal necessary permissions
3. **Defense in Depth**: Multiple security layers
4. **Secure by Default**: Security built-in, not bolted-on
5. **Continuous Monitoring**: Real-time threat detection

## 🔒 Authentication & Authorization

### JWT Implementation
```python
# src/infrastructure/security/auth.py
from datetime import datetime, timedelta
from typing import Optional, Dict
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, HTTPBearer
import secrets

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")

# HTTP Bearer scheme for API keys
security = HTTPBearer()

class AuthService:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire = timedelta(minutes=30)
        self.refresh_token_expire = timedelta(days=7)
    
    def create_access_token(self, data: Dict, expires_delta: Optional[timedelta] = None):
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + self.access_token_expire
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict):
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + self.refresh_token_expire
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
            "jti": secrets.token_urlsafe(32)  # Unique token ID
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            if payload.get("type") != token_type:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)

# Dependency for getting current user
async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict:
    """Get current authenticated user"""
    auth_service = AuthService(settings.security.secret_key)
    payload = auth_service.verify_token(token)
    
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )
    
    # Get user from database
    user = await user_repository.get_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return user
```

### Role-Based Access Control (RBAC)
```python
# src/infrastructure/security/rbac.py
from enum import Enum
from typing import List, Set
from functools import wraps

class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    API_CLIENT = "api_client"

class Permission(str, Enum):
    # Document permissions
    DOCUMENT_READ = "document:read"
    DOCUMENT_WRITE = "document:write"
    DOCUMENT_DELETE = "document:delete"
    
    # Search permissions
    SEARCH_BASIC = "search:basic"
    SEARCH_ADVANCED = "search:advanced"
    SEARCH_ADMIN = "search:admin"
    
    # Admin permissions
    USER_MANAGE = "user:manage"
    SYSTEM_CONFIG = "system:config"
    METRICS_VIEW = "metrics:view"

# Role-Permission mapping
ROLE_PERMISSIONS = {
    Role.ADMIN: {
        Permission.DOCUMENT_READ,
        Permission.DOCUMENT_WRITE,
        Permission.DOCUMENT_DELETE,
        Permission.SEARCH_BASIC,
        Permission.SEARCH_ADVANCED,
        Permission.SEARCH_ADMIN,
        Permission.USER_MANAGE,
        Permission.SYSTEM_CONFIG,
        Permission.METRICS_VIEW
    },
    Role.USER: {
        Permission.DOCUMENT_READ,
        Permission.DOCUMENT_WRITE,
        Permission.SEARCH_BASIC,
        Permission.SEARCH_ADVANCED
    },
    Role.VIEWER: {
        Permission.DOCUMENT_READ,
        Permission.SEARCH_BASIC
    },
    Role.API_CLIENT: {
        Permission.DOCUMENT_READ,
        Permission.SEARCH_BASIC,
        Permission.SEARCH_ADVANCED
    }
}

class RBACService:
    @staticmethod
    def has_permission(user_role: Role, required_permission: Permission) -> bool:
        """Check if role has required permission"""
        role_permissions = ROLE_PERMISSIONS.get(user_role, set())
        return required_permission in role_permissions
    
    @staticmethod
    def get_permissions(user_role: Role) -> Set[Permission]:
        """Get all permissions for a role"""
        return ROLE_PERMISSIONS.get(user_role, set())

# Permission decorator
def require_permission(permission: Permission):
    """Decorator to check permissions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, current_user: Dict = Depends(get_current_user), **kwargs):
            user_role = Role(current_user.get("role", Role.VIEWER))
            
            if not RBACService.has_permission(user_role, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied. Required: {permission.value}"
                )
            
            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator

# Usage example
@router.post("/documents")
@require_permission(Permission.DOCUMENT_WRITE)
async def create_document(
    document: DocumentCreate,
    current_user: Dict = Depends(get_current_user)
):
    # Only users with DOCUMENT_WRITE permission can access
    pass
```

### API Key Management
```python
# src/infrastructure/security/api_keys.py
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional

class APIKeyService:
    def __init__(self, repository: APIKeyRepository):
        self.repository = repository
    
    async def create_api_key(
        self,
        user_id: str,
        name: str,
        expires_in_days: Optional[int] = None,
        permissions: List[Permission] = None
    ) -> Dict[str, str]:
        """Create new API key"""
        # Generate secure random key
        raw_key = secrets.token_urlsafe(32)
        
        # Hash the key for storage
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        # Store in database
        api_key = await self.repository.create({
            "user_id": user_id,
            "name": name,
            "key_hash": key_hash,
            "permissions": permissions or [],
            "expires_at": expires_at,
            "last_used_at": None,
            "created_at": datetime.utcnow()
        })
        
        # Return the raw key (only shown once)
        return {
            "id": api_key.id,
            "key": raw_key,
            "name": name,
            "expires_at": expires_at
        }
    
    async def verify_api_key(self, raw_key: str) -> Optional[Dict]:
        """Verify API key and return associated data"""
        # Hash the provided key
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # Look up in database
        api_key = await self.repository.get_by_hash(key_hash)
        
        if not api_key:
            return None
        
        # Check expiration
        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
            return None
        
        # Update last used timestamp
        await self.repository.update_last_used(api_key.id)
        
        return {
            "user_id": api_key.user_id,
            "permissions": api_key.permissions,
            "name": api_key.name
        }

# API Key authentication dependency
async def get_api_key_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict:
    """Authenticate using API key"""
    api_key = credentials.credentials
    
    api_key_service = APIKeyService(api_key_repository)
    key_data = await api_key_service.verify_api_key(api_key)
    
    if not key_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return key_data
```

## 🛡️ Input Validation & Sanitization

### Request Validation
```python
# src/presentation/api/validators.py
from pydantic import BaseModel, Field, validator
import re
from typing import List, Optional
import bleach

class SearchQueryValidator(BaseModel):
    """Validate and sanitize search queries"""
    
    text: str = Field(..., min_length=1, max_length=1000)
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    max_results: int = Field(10, ge=1, le=100)
    
    @validator('text')
    def sanitize_query_text(cls, v):
        """Remove potentially harmful characters"""
        # Remove null bytes
        v = v.replace('\x00', '')
        
        # Remove control characters
        v = ''.join(char for char in v if ord(char) >= 32 or char in '\n\r\t')
        
        # Prevent SQL injection patterns
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE|ALTER)\b)",
            r"(--|#|/\*|\*/)",
            r"(\bOR\b.*=.*)",
            r"(\bAND\b.*=.*)"
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Invalid characters in query")
        
        return v.strip()
    
    @validator('filters')
    def validate_filters(cls, v):
        """Validate filter structure"""
        allowed_keys = {'category', 'date_from', 'date_to', 'language', 'tags'}
        
        for key in v.keys():
            if key not in allowed_keys:
                raise ValueError(f"Invalid filter key: {key}")
        
        return v

class DocumentUploadValidator(BaseModel):
    """Validate document uploads"""
    
    file_type: str
    file_size: int
    metadata: Dict[str, Any]
    
    @validator('file_type')
    def validate_file_type(cls, v):
        """Check allowed file types"""
        allowed_types = {
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'text/plain',
            'text/markdown',
            'application/json'
        }
        
        if v not in allowed_types:
            raise ValueError(f"File type {v} not allowed")
        
        return v
    
    @validator('file_size')
    def validate_file_size(cls, v):
        """Check file size limits"""
        max_size = 100 * 1024 * 1024  # 100MB
        
        if v > max_size:
            raise ValueError(f"File size exceeds maximum of {max_size} bytes")
        
        return v

# HTML content sanitization
class ContentSanitizer:
    @staticmethod
    def sanitize_html(content: str) -> str:
        """Sanitize HTML content"""
        allowed_tags = [
            'p', 'br', 'span', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'strong', 'em', 'u', 'strike', 'ul', 'ol', 'li', 'a', 'img',
            'table', 'thead', 'tbody', 'tr', 'th', 'td', 'blockquote', 'code', 'pre'
        ]
        
        allowed_attributes = {
            'a': ['href', 'title'],
            'img': ['src', 'alt', 'width', 'height'],
            'td': ['colspan', 'rowspan'],
            'th': ['colspan', 'rowspan']
        }
        
        # Clean HTML
        cleaned = bleach.clean(
            content,
            tags=allowed_tags,
            attributes=allowed_attributes,
            strip=True
        )
        
        # Additional XSS prevention
        cleaned = cleaned.replace('javascript:', '')
        cleaned = cleaned.replace('vbscript:', '')
        cleaned = cleaned.replace('onload=', '')
        cleaned = cleaned.replace('onerror=', '')
        
        return cleaned
```

### File Upload Security
```python
# src/infrastructure/security/file_security.py
import magic
import hashlib
from pathlib import Path
import tempfile
import subprocess

class FileSecurityService:
    def __init__(self):
        self.allowed_mimes = {
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'text/plain',
            'text/markdown',
            'image/jpeg',
            'image/png'
        }
        self.max_file_size = 100 * 1024 * 1024  # 100MB
    
    async def validate_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Comprehensive file validation"""
        
        # Check file size
        if len(file_content) > self.max_file_size:
            raise ValueError("File size exceeds maximum allowed")
        
        # Check MIME type using python-magic
        mime_type = magic.from_buffer(file_content, mime=True)
        if mime_type not in self.allowed_mimes:
            raise ValueError(f"File type {mime_type} not allowed")
        
        # Check file extension
        file_ext = Path(filename).suffix.lower()
        allowed_extensions = {'.pdf', '.docx', '.txt', '.md', '.jpg', '.jpeg', '.png'}
        if file_ext not in allowed_extensions:
            raise ValueError(f"File extension {file_ext} not allowed")
        
        # Calculate file hash
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Virus scan (if ClamAV is available)
        virus_scan_result = await self._scan_for_viruses(file_content)
        if not virus_scan_result['clean']:
            raise ValueError(f"File failed virus scan: {virus_scan_result['message']}")
        
        return {
            "mime_type": mime_type,
            "file_hash": file_hash,
            "file_size": len(file_content),
            "virus_scan": virus_scan_result
        }
    
    async def _scan_for_viruses(self, file_content: bytes) -> Dict[str, Any]:
        """Scan file for viruses using ClamAV"""
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(file_content)
                tmp_file.flush()
                
                # Run ClamAV scan
                result = subprocess.run(
                    ['clamscan', '--no-summary', tmp_file.name],
                    capture_output=True,
                    text=True
                )
                
                # Clean up
                Path(tmp_file.name).unlink()
                
                if result.returncode == 0:
                    return {"clean": True, "message": "No threats found"}
                else:
                    return {"clean": False, "message": result.stdout}
                    
        except FileNotFoundError:
            # ClamAV not installed
            return {"clean": True, "message": "Virus scanning not available"}
```

## 🔐 Data Encryption

### Encryption at Rest
```python
# src/infrastructure/security/encryption.py
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class EncryptionService:
    def __init__(self, master_key: str):
        # Derive encryption key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'stable_salt',  # In production, use proper salt management
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        self.cipher = Fernet(key)
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        encrypted = self.cipher.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        decoded = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = self.cipher.decrypt(decoded)
        return decrypted.decode()
    
    def encrypt_file(self, file_path: str, output_path: str):
        """Encrypt file contents"""
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        encrypted_data = self.cipher.encrypt(file_data)
        
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
    
    def decrypt_file(self, encrypted_path: str, output_path: str):
        """Decrypt file contents"""
        with open(encrypted_path, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = self.cipher.decrypt(encrypted_data)
        
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)

# Database field encryption
from sqlalchemy import TypeDecorator, String

class EncryptedType(TypeDecorator):
    """SQLAlchemy type for encrypted fields"""
    impl = String
    
    def __init__(self, encryption_service: EncryptionService, *args, **kwargs):
        self.encryption_service = encryption_service
        super().__init__(*args, **kwargs)
    
    def process_bind_param(self, value, dialect):
        """Encrypt before storing"""
        if value is not None:
            return self.encryption_service.encrypt_data(value)
        return value
    
    def process_result_value(self, value, dialect):
        """Decrypt after retrieving"""
        if value is not None:
            return self.encryption_service.decrypt_data(value)
        return value
```

### Secure Communication
```python
# src/infrastructure/security/tls.py
import ssl
import certifi
from aiohttp import ClientSession, TCPConnector

class SecureHTTPClient:
    def __init__(self):
        # Create SSL context with strict settings
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.ssl_context.check_hostname = True
        self.ssl_context.verify_mode = ssl.CERT_REQUIRED
        self.ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        
        # Disable weak ciphers
        self.ssl_context.set_ciphers(
            'ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS'
        )
    
    async def create_session(self) -> ClientSession:
        """Create secure HTTP session"""
        connector = TCPConnector(
            ssl=self.ssl_context,
            limit=100,
            limit_per_host=30
        )
        
        return ClientSession(
            connector=connector,
            headers={
                'User-Agent': 'RAG-System/1.0',
                'Accept': 'application/json'
            }
        )
```

## 🚨 Security Headers

### FastAPI Security Headers Middleware
```python
# src/presentation/middleware/security_headers.py
from fastapi import Request
from fastapi.responses import Response
import uuid

class SecurityHeadersMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, request: Request, call_next):
        # Generate request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self' wss: https:;"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )
        
        # Remove sensitive headers
        response.headers.pop("Server", None)
        response.headers.pop("X-Powered-By", None)
        
        return response
```

## 🔒 Secrets Management

### Environment Variable Security
```python
# src/infrastructure/security/secrets.py
import os
from typing import Optional
import boto3
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

class SecretsManager:
    def __init__(self, provider: str = "env"):
        self.provider = provider
        self._client = self._init_client()
    
    def _init_client(self):
        """Initialize secrets provider client"""
        if self.provider == "aws":
            return boto3.client('secretsmanager')
        elif self.provider == "azure":
            credential = DefaultAzureCredential()
            vault_url = os.getenv("AZURE_KEY_VAULT_URL")
            return SecretClient(vault_url=vault_url, credential=credential)
        else:
            return None  # Use environment variables
    
    def get_secret(self, secret_name: str) -> Optional[str]:
        """Retrieve secret value"""
        if self.provider == "env":
            return os.getenv(secret_name)
        
        elif self.provider == "aws":
            try:
                response = self._client.get_secret_value(SecretId=secret_name)
                return response['SecretString']
            except Exception as e:
                logger.error(f"Failed to retrieve AWS secret: {e}")
                return None
        
        elif self.provider == "azure":
            try:
                secret = self._client.get_secret(secret_name)
                return secret.value
            except Exception as e:
                logger.error(f"Failed to retrieve Azure secret: {e}")
                return None
    
    def get_database_url(self) -> str:
        """Get database connection string"""
        if self.provider == "env":
            return os.getenv("DATABASE_URL")
        else:
            # Construct from individual secrets
            host = self.get_secret("db_host")
            port = self.get_secret("db_port")
            user = self.get_secret("db_user")
            password = self.get_secret("db_password")
            database = self.get_secret("db_name")
            
            return f"postgresql://{user}:{password}@{host}:{port}/{database}"

# Kubernetes secrets integration
apiVersion: v1
kind: Secret
metadata:
  name: rag-secrets
  namespace: rag-system
type: Opaque
data:
  database-url: <base64-encoded-value>
  openai-api-key: <base64-encoded-value>
  jwt-secret-key: <base64-encoded-value>
```

## 🛡️ Rate Limiting & DDoS Protection

### Advanced Rate Limiting
```python
# src/infrastructure/security/rate_limiting.py
from typing import Callable, Optional
import time
import asyncio
from collections import defaultdict
import redis.asyncio as redis

class AdvancedRateLimiter:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.limits = {
            "default": {"requests": 60, "window": 60},  # 60 req/min
            "search": {"requests": 30, "window": 60},   # 30 searches/min
            "upload": {"requests": 10, "window": 3600}, # 10 uploads/hour
            "expensive": {"requests": 5, "window": 60}  # 5 expensive ops/min
        }
    
    async def check_rate_limit(
        self,
        key: str,
        limit_type: str = "default",
        cost: int = 1
    ) -> tuple[bool, dict]:
        """Check if request is within rate limit"""
        
        limit_config = self.limits.get(limit_type, self.limits["default"])
        window = limit_config["window"]
        max_requests = limit_config["requests"]
        
        # Use sliding window counter
        now = time.time()
        window_start = now - window
        
        # Redis key for this limit
        redis_key = f"rate_limit:{limit_type}:{key}"
        
        # Remove old entries
        await self.redis.zremrangebyscore(redis_key, 0, window_start)
        
        # Count current requests in window
        current_count = await self.redis.zcard(redis_key)
        
        if current_count + cost > max_requests:
            # Rate limit exceeded
            ttl = await self.redis.ttl(redis_key)
            return False, {
                "limit": max_requests,
                "remaining": max(0, max_requests - current_count),
                "reset": int(now + ttl)
            }
        
        # Add current request
        await self.redis.zadd(redis_key, {f"{now}:{cost}": now})
        await self.redis.expire(redis_key, window)
        
        return True, {
            "limit": max_requests,
            "remaining": max_requests - current_count - cost,
            "reset": int(now + window)
        }

# Rate limiting middleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

def create_rate_limiter():
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=["100 per minute"],
        storage_uri=settings.redis.url
    )
    
    return limiter

# Custom rate limit decorator
def rate_limit(limit_type: str = "default", cost: int = 1):
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Get client identifier
            client_id = get_remote_address(request)
            if hasattr(request.state, "user"):
                client_id = f"user:{request.state.user.id}"
            
            # Check rate limit
            limiter = request.app.state.rate_limiter
            allowed, headers = await limiter.check_rate_limit(
                client_id, limit_type, cost
            )
            
            # Add rate limit headers
            response = await func(request, *args, **kwargs)
            response.headers["X-RateLimit-Limit"] = str(headers["limit"])
            response.headers["X-RateLimit-Remaining"] = str(headers["remaining"])
            response.headers["X-RateLimit-Reset"] = str(headers["reset"])
            
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded"
                )
            
            return response
        return wrapper
    return decorator
```

## 🔍 Security Monitoring & Auditing

### Audit Logging
```python
# src/infrastructure/security/audit.py
from datetime import datetime
from typing import Dict, Any
import json

class AuditLogger:
    def __init__(self, repository: AuditLogRepository):
        self.repository = repository
    
    async def log_event(
        self,
        event_type: str,
        user_id: Optional[str],
        resource_type: str,
        resource_id: Optional[str],
        action: str,
        outcome: str,
        metadata: Dict[str, Any] = None,
        ip_address: Optional[str] = None
    ):
        """Log security-relevant event"""
        
        audit_entry = {
            "timestamp": datetime.utcnow(),
            "event_type": event_type,
            "user_id": user_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "action": action,
            "outcome": outcome,
            "metadata": metadata or {},
            "ip_address": ip_address
        }
        
        await self.repository.create(audit_entry)
        
        # Log to structured logging as well
        logger.info(
            "audit_event",
            **audit_entry
        )
    
    async def log_authentication(
        self,
        user_id: Optional[str],
        action: str,  # login, logout, failed_login
        success: bool,
        ip_address: str,
        user_agent: str
    ):
        """Log authentication events"""
        await self.log_event(
            event_type="authentication",
            user_id=user_id,
            resource_type="auth",
            resource_id=None,
            action=action,
            outcome="success" if success else "failure",
            metadata={
                "user_agent": user_agent,
                "method": "password"  # or "api_key", "oauth"
            },
            ip_address=ip_address
        )
    
    async def log_data_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,  # read, write, delete
        filters: Dict[str, Any] = None
    ):
        """Log data access events"""
        await self.log_event(
            event_type="data_access",
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            outcome="success",
            metadata={"filters": filters} if filters else {}
        )

# Audit middleware
class AuditMiddleware:
    def __init__(self, app, audit_logger: AuditLogger):
        self.app = app
        self.audit_logger = audit_logger
    
    async def __call__(self, request: Request, call_next):
        # Capture request details
        start_time = time.time()
        
        # Get user info if authenticated
        user_id = None
        if hasattr(request.state, "user"):
            user_id = request.state.user.get("id")
        
        # Process request
        response = await call_next(request)
        
        # Log API access
        if response.status_code < 400:
            await self.audit_logger.log_event(
                event_type="api_access",
                user_id=user_id,
                resource_type="api",
                resource_id=request.url.path,
                action=request.method,
                outcome="success",
                metadata={
                    "status_code": response.status_code,
                    "duration_ms": (time.time() - start_time) * 1000
                },
                ip_address=request.client.host if request.client else None
            )
        
        return response
```

## 🔐 Security Scanning & Compliance

### Dependency Scanning
```yaml
# .github/workflows/security.yml
name: Security Scan

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * *'  # Daily scan

jobs:
  dependency-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run Safety check
      run: |
        pip install safety
        safety check --json > safety-report.json
    
    - name: Run Bandit security scan
      run: |
        pip install bandit
        bandit -r src/ -f json -o bandit-report.json
    
    - name: OWASP Dependency Check
      uses: dependency-check/Dependency-Check_Action@main
      with:
        project: 'rag-system'
        path: '.'
        format: 'JSON'
    
    - name: Upload results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'reports'

  container-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'rag-system:latest'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
```

### Security Checklist
```python
# scripts/security_check.py
#!/usr/bin/env python3
"""Security compliance checker"""

import subprocess
import json
from pathlib import Path

class SecurityChecker:
    def __init__(self):
        self.checks = []
    
    def check_secrets_in_code(self):
        """Check for hardcoded secrets"""
        result = subprocess.run(
            ["git", "secrets", "--scan"],
            capture_output=True,
            text=True
        )
        
        self.checks.append({
            "name": "No hardcoded secrets",
            "passed": result.returncode == 0,
            "details": result.stdout
        })
    
    def check_dependencies(self):
        """Check for vulnerable dependencies"""
        result = subprocess.run(
            ["safety", "check", "--json"],
            capture_output=True,
            text=True
        )
        
        vulnerabilities = json.loads(result.stdout) if result.stdout else []
        
        self.checks.append({
            "name": "No vulnerable dependencies",
            "passed": len(vulnerabilities) == 0,
            "details": f"Found {len(vulnerabilities)} vulnerabilities"
        })
    
    def check_docker_security(self):
        """Check Docker security best practices"""
        dockerfile = Path("Dockerfile")
        if dockerfile.exists():
            content = dockerfile.read_text()
            
            checks = {
                "Non-root user": "USER" in content,
                "No sudo": "sudo" not in content,
                "Health check": "HEALTHCHECK" in content,
                "No latest tags": ":latest" not in content
            }
            
            passed = all(checks.values())
            
            self.checks.append({
                "name": "Docker security",
                "passed": passed,
                "details": checks
            })
    
    def generate_report(self):
        """Generate security report"""
        passed = sum(1 for check in self.checks if check["passed"])
        total = len(self.checks)
        
        print(f"Security Check Report: {passed}/{total} passed")
        print("-" * 50)
        
        for check in self.checks:
            status = "✓" if check["passed"] else "✗"
            print(f"{status} {check['name']}")
            if not check["passed"]:
                print(f"  Details: {check['details']}")

if __name__ == "__main__":
    checker = SecurityChecker()
    checker.check_secrets_in_code()
    checker.check_dependencies()
    checker.check_docker_security()
    checker.generate_report()
```

## 🚀 Security Best Practices Summary

### Development
1. **Never commit secrets** - Use environment variables or secret management
2. **Validate all inputs** - Both client and server side
3. **Use parameterized queries** - Prevent SQL injection
4. **Implement proper error handling** - Don't expose internal details
5. **Keep dependencies updated** - Regular security patches

### Deployment
1. **Use HTTPS everywhere** - Encrypt data in transit
2. **Implement rate limiting** - Prevent abuse
3. **Enable security headers** - Protect against common attacks
4. **Run as non-root user** - Minimize container privileges
5. **Network segmentation** - Isolate services

### Operations
1. **Monitor security events** - Real-time threat detection
2. **Regular security audits** - Compliance checks
3. **Incident response plan** - Be prepared for breaches
4. **Backup encryption keys** - Disaster recovery
5. **Access reviews** - Regular permission audits