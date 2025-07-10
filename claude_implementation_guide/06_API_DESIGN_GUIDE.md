# API Design Guide - RAG Comprehensive System

## 🎯 API Design Principles

### RESTful Design
- Use proper HTTP methods (GET, POST, PUT, DELETE)
- Resource-based URLs
- Stateless operations
- HATEOAS where applicable

### Consistency
- Consistent naming conventions
- Standardized error responses
- Unified pagination format
- Common filtering syntax

### Versioning
- URL-based versioning (/api/v1/)
- Backward compatibility guarantees
- Deprecation policies

## 📚 API Structure

### Base URL Pattern
```
https://api.rag-system.com/api/v1/{resource}
```

### Authentication
```http
Authorization: Bearer <jwt-token>
X-API-Key: <api-key> (alternative)
```

## 🔗 Core Endpoints

### 1. Document Management

#### Upload Document
```http
POST /api/v1/documents
Content-Type: multipart/form-data

Request:
{
  "file": <binary>,
  "metadata": {
    "title": "Machine Learning Guide",
    "category": "technical",
    "tags": ["ml", "ai", "deep-learning"],
    "language": "en",
    "custom_fields": {
      "department": "engineering",
      "version": "1.0"
    }
  },
  "processing_options": {
    "chunking_strategy": "semantic",
    "chunk_size": 1000,
    "enable_ocr": true,
    "extract_images": true
  }
}

Response:
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "estimated_completion": "2024-01-15T10:05:00Z",
  "created_at": "2024-01-15T10:00:00Z",
  "_links": {
    "self": "/api/v1/documents/550e8400-e29b-41d4-a716-446655440000",
    "status": "/api/v1/documents/550e8400-e29b-41d4-a716-446655440000/status"
  }
}
```

#### Get Document
```http
GET /api/v1/documents/{document_id}

Response:
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "Machine Learning Guide",
  "content": "...",
  "metadata": {
    "title": "Machine Learning Guide",
    "category": "technical",
    "tags": ["ml", "ai", "deep-learning"],
    "file_type": "pdf",
    "file_size_bytes": 2048576,
    "page_count": 42,
    "language": "en",
    "created_date": "2024-01-10T08:00:00Z"
  },
  "processing_status": "completed",
  "chunks_count": 45,
  "embedding_model": "text-embedding-3-large",
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:05:00Z",
  "_links": {
    "self": "/api/v1/documents/550e8400-e29b-41d4-a716-446655440000",
    "chunks": "/api/v1/documents/550e8400-e29b-41d4-a716-446655440000/chunks",
    "download": "/api/v1/documents/550e8400-e29b-41d4-a716-446655440000/download"
  }
}
```

#### List Documents
```http
GET /api/v1/documents?page=1&size=20&category=technical&sort=created_at:desc

Response:
{
  "items": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "title": "Machine Learning Guide",
      "category": "technical",
      "created_at": "2024-01-15T10:00:00Z",
      "status": "completed"
    }
  ],
  "pagination": {
    "page": 1,
    "size": 20,
    "total_items": 156,
    "total_pages": 8
  },
  "_links": {
    "self": "/api/v1/documents?page=1&size=20",
    "next": "/api/v1/documents?page=2&size=20",
    "last": "/api/v1/documents?page=8&size=20"
  }
}
```

### 2. Search & Retrieval

#### Basic Search
```http
POST /api/v1/search
Content-Type: application/json

Request:
{
  "query": "What are the main types of machine learning?",
  "filters": {
    "category": ["technical", "educational"],
    "created_after": "2024-01-01",
    "language": "en"
  },
  "options": {
    "max_results": 10,
    "similarity_threshold": 0.7,
    "include_metadata": true,
    "highlight_matches": true
  }
}

Response:
{
  "query_id": "q_123456",
  "results": [
    {
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "chunk_id": "chunk_789",
      "content": "There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning...",
      "score": 0.92,
      "metadata": {
        "source": "ml-guide.pdf",
        "page": 5,
        "section": "Introduction"
      },
      "highlights": [
        {
          "start": 10,
          "end": 35,
          "text": "three main types of machine learning"
        }
      ]
    }
  ],
  "total_results": 10,
  "processing_time_ms": 145,
  "strategy_used": "hybrid"
}
```

#### Advanced RAG Search
```http
POST /api/v1/search/advanced
Content-Type: application/json

Request:
{
  "query": "Compare transformer architecture with LSTM for NLP tasks",
  "strategy": "self_rag",
  "retrieval_config": {
    "initial_k": 50,
    "rerank_k": 10,
    "use_mmr": true,
    "diversity_lambda": 0.5,
    "enable_graph_expansion": true,
    "max_graph_hops": 2
  },
  "generation_config": {
    "temperature": 0.7,
    "max_tokens": 500,
    "include_citations": true,
    "format": "markdown"
  },
  "self_rag_config": {
    "max_iterations": 3,
    "confidence_threshold": 0.8,
    "enable_correction": true
  }
}

Response:
{
  "query_id": "q_789012",
  "answer": "## Transformer vs LSTM for NLP Tasks\n\nTransformers have largely superseded LSTMs...",
  "confidence_score": 0.91,
  "citations": [
    {
      "document_id": "doc_123",
      "chunk_id": "chunk_456",
      "relevance": 0.94,
      "text": "Attention is All You Need paper..."
    }
  ],
  "retrieval_metadata": {
    "strategy": "self_rag",
    "iterations": 2,
    "documents_retrieved": 45,
    "documents_used": 8,
    "reformulations": [
      "transformer LSTM comparison NLP",
      "attention mechanism vs recurrent networks"
    ]
  },
  "processing_time_ms": 2340
}
```

#### Multimodal Search
```http
POST /api/v1/search/multimodal
Content-Type: multipart/form-data

Request:
{
  "text_query": "diagrams showing neural network architectures",
  "image_query": <binary image data>,
  "search_mode": "hybrid", // text_only, image_only, hybrid
  "options": {
    "max_results": 20,
    "include_similar_images": true,
    "cross_modal_weight": 0.6
  }
}

Response:
{
  "results": [
    {
      "type": "image",
      "document_id": "doc_456",
      "image_id": "img_789",
      "caption": "Transformer architecture diagram",
      "similarity_score": 0.89,
      "image_url": "/api/v1/images/img_789",
      "thumbnail_url": "/api/v1/images/img_789/thumbnail",
      "source_page": 12
    },
    {
      "type": "text",
      "document_id": "doc_123",
      "chunk_id": "chunk_234",
      "content": "Figure 3 shows the complete transformer architecture...",
      "similarity_score": 0.85,
      "associated_images": ["img_790", "img_791"]
    }
  ]
}
```

### 3. Real-time Operations

#### WebSocket Connection
```javascript
// WebSocket endpoint: wss://api.rag-system.com/ws

// Connection
const ws = new WebSocket('wss://api.rag-system.com/ws');

// Authentication
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'Bearer <jwt-token>'
  }));
};

// Streaming search
ws.send(JSON.stringify({
  type: 'search_stream',
  data: {
    query: 'Explain quantum computing',
    stream_response: true
  }
}));

// Receive chunks
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  switch(message.type) {
    case 'retrieval_progress':
      console.log(`Retrieved ${message.data.documents} documents`);
      break;
    
    case 'response_chunk':
      // Append to UI
      appendToResponse(message.data.text);
      break;
    
    case 'search_complete':
      console.log('Search completed', message.data);
      break;
  }
};
```

### 4. Batch Operations

#### Batch Document Upload
```http
POST /api/v1/documents/batch
Content-Type: application/json

Request:
{
  "documents": [
    {
      "url": "https://example.com/doc1.pdf",
      "metadata": {"category": "technical"}
    },
    {
      "base64_content": "JVBERi0xLjQKJeLj...",
      "filename": "doc2.pdf",
      "metadata": {"category": "research"}
    }
  ],
  "processing_options": {
    "parallel_processing": true,
    "priority": "high"
  }
}

Response:
{
  "batch_id": "batch_123456",
  "documents": [
    {
      "index": 0,
      "document_id": "doc_789",
      "status": "queued"
    },
    {
      "index": 1,
      "document_id": "doc_790",
      "status": "queued"
    }
  ],
  "_links": {
    "status": "/api/v1/batches/batch_123456/status"
  }
}
```

#### Batch Status Check
```http
GET /api/v1/batches/{batch_id}/status

Response:
{
  "batch_id": "batch_123456",
  "status": "processing",
  "progress": {
    "total": 10,
    "completed": 7,
    "failed": 1,
    "processing": 2
  },
  "documents": [
    {
      "document_id": "doc_789",
      "status": "completed",
      "processing_time_ms": 2345
    },
    {
      "document_id": "doc_790",
      "status": "failed",
      "error": "Unsupported file format"
    }
  ]
}
```

### 5. Analytics & Monitoring

#### Query Analytics
```http
GET /api/v1/analytics/queries?start_date=2024-01-01&end_date=2024-01-31

Response:
{
  "period": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-31T23:59:59Z"
  },
  "summary": {
    "total_queries": 15234,
    "unique_users": 342,
    "avg_response_time_ms": 234,
    "success_rate": 0.98
  },
  "top_queries": [
    {
      "query": "machine learning basics",
      "count": 234,
      "avg_confidence": 0.89
    }
  ],
  "strategies_used": {
    "semantic": 5234,
    "hybrid": 7890,
    "self_rag": 2110
  },
  "performance_metrics": {
    "p50_latency_ms": 180,
    "p95_latency_ms": 450,
    "p99_latency_ms": 890
  }
}
```

## 🔒 Error Handling

### Standard Error Response
```json
{
  "error": {
    "code": "DOCUMENT_NOT_FOUND",
    "message": "The requested document was not found",
    "details": {
      "document_id": "550e8400-e29b-41d4-a716-446655440000"
    },
    "timestamp": "2024-01-15T10:00:00Z",
    "request_id": "req_123456"
  }
}
```

### Error Codes
| Code | HTTP Status | Description |
|------|-------------|-------------|
| INVALID_REQUEST | 400 | Malformed request |
| AUTHENTICATION_REQUIRED | 401 | Missing or invalid auth |
| FORBIDDEN | 403 | Insufficient permissions |
| NOT_FOUND | 404 | Resource not found |
| RATE_LIMIT_EXCEEDED | 429 | Too many requests |
| INTERNAL_ERROR | 500 | Server error |
| SERVICE_UNAVAILABLE | 503 | Temporary unavailable |

## 🔄 Pagination

### Cursor-based Pagination
```http
GET /api/v1/documents?cursor=eyJpZCI6MTIzfQ&size=20

Response:
{
  "items": [...],
  "cursors": {
    "next": "eyJpZCI6MTQzfQ",
    "previous": "eyJpZCI6MTAzfQ"
  },
  "has_more": true
}
```

### Offset-based Pagination
```http
GET /api/v1/documents?page=2&size=20

Response:
{
  "items": [...],
  "pagination": {
    "page": 2,
    "size": 20,
    "total_items": 156,
    "total_pages": 8
  }
}
```

## 🔍 Filtering & Sorting

### Filter Syntax
```http
GET /api/v1/documents?filter[category]=technical&filter[created_at][gte]=2024-01-01

# Complex filters
GET /api/v1/documents?filter[or][0][category]=technical&filter[or][1][category]=research
```

### Sort Syntax
```http
GET /api/v1/documents?sort=created_at:desc,title:asc
```

## 📊 Response Formats

### Content Negotiation
```http
# JSON (default)
Accept: application/json

# CSV export
Accept: text/csv

# XML
Accept: application/xml
```

### Field Selection
```http
GET /api/v1/documents?fields=id,title,metadata.category
```

## 🚀 Performance Headers

### Caching
```http
Cache-Control: private, max-age=300
ETag: "33a64df551425fcc55e4d42a148795d9f25f89d4"
Last-Modified: Wed, 21 Oct 2024 07:28:00 GMT
```

### Rate Limiting
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1672531200
```

### Request ID
```http
X-Request-ID: req_123456789
```

## 🧪 API Testing

### Health Check
```http
GET /api/v1/health

Response:
{
  "status": "healthy",
  "version": "1.2.3",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "ml_services": "healthy"
  }
}
```

### Echo Test
```http
POST /api/v1/test/echo

Request:
{
  "message": "Hello, World!"
}

Response:
{
  "echo": "Hello, World!",
  "timestamp": "2024-01-15T10:00:00Z"
}
```

## 📚 SDK Examples

### Python SDK
```python
from rag_system import RAGClient

client = RAGClient(api_key="your-api-key")

# Upload document
doc = client.documents.upload(
    file_path="document.pdf",
    metadata={"category": "technical"}
)

# Search
results = client.search(
    query="machine learning",
    strategy="hybrid",
    max_results=10
)

# Streaming search
async for chunk in client.search_stream(query="explain AI"):
    print(chunk.text)
```

### JavaScript SDK
```javascript
import { RAGClient } from '@rag-system/sdk';

const client = new RAGClient({ apiKey: 'your-api-key' });

// Upload document
const doc = await client.documents.upload({
  file: fileBuffer,
  metadata: { category: 'technical' }
});

// Search with streaming
const stream = await client.search({
  query: 'machine learning',
  stream: true
});

for await (const chunk of stream) {
  console.log(chunk.text);
}
```

## 🔐 Security Best Practices

### API Key Management
- Rotate keys regularly
- Use different keys for different environments
- Implement key scoping

### Request Signing
```http
X-Signature: HMAC-SHA256(request-body + timestamp)
X-Timestamp: 1672531200
```

### CORS Configuration
```javascript
// Allowed origins
Access-Control-Allow-Origin: https://app.rag-system.com
Access-Control-Allow-Methods: GET, POST, PUT, DELETE
Access-Control-Allow-Headers: Content-Type, Authorization
```