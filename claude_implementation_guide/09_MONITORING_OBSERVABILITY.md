# Monitoring & Observability Guide - RAG Comprehensive System

## 🎯 Observability Strategy

### Three Pillars of Observability
1. **Metrics**: Quantitative data about system performance
2. **Logs**: Detailed event records for debugging
3. **Traces**: Request flow through distributed system

### Key Principles
- **Proactive Monitoring**: Detect issues before users report them
- **End-to-End Visibility**: Track requests across all services
- **Actionable Alerts**: Reduce noise, focus on business impact
- **Data-Driven Decisions**: Use metrics to guide optimization

## 📊 Metrics Implementation

### Application Metrics
```python
# src/infrastructure/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info, Summary
from functools import wraps
import time
from typing import Callable

# Request metrics
http_requests_total = Counter(
    'rag_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

http_request_duration_seconds = Histogram(
    'rag_http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
)

active_requests = Gauge(
    'rag_active_requests',
    'Number of active requests',
    ['endpoint']
)

# Business metrics
documents_processed_total = Counter(
    'rag_documents_processed_total',
    'Total documents processed',
    ['file_type', 'status', 'processing_strategy']
)

search_queries_total = Counter(
    'rag_search_queries_total',
    'Total search queries',
    ['strategy', 'user_type']
)

retrieval_latency_seconds = Histogram(
    'rag_retrieval_latency_seconds',
    'Retrieval latency by strategy',
    ['strategy', 'query_complexity'],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

generation_latency_seconds = Histogram(
    'rag_generation_latency_seconds',
    'LLM generation latency',
    ['model', 'prompt_size_category']
)

# Cache metrics
cache_hits_total = Counter(
    'rag_cache_hits_total',
    'Cache hits',
    ['cache_type', 'operation']
)

cache_misses_total = Counter(
    'rag_cache_misses_total',
    'Cache misses',
    ['cache_type', 'operation']
)

# ML model metrics
embedding_generation_duration = Summary(
    'rag_embedding_generation_duration_seconds',
    'Time to generate embeddings',
    ['model', 'batch_size_category']
)

reranking_scores = Histogram(
    'rag_reranking_scores',
    'Distribution of reranking scores',
    ['model', 'query_type'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# System metrics
database_connection_pool_size = Gauge(
    'rag_database_connection_pool_size',
    'Database connection pool size',
    ['pool_name']
)

background_jobs_queued = Gauge(
    'rag_background_jobs_queued',
    'Number of queued background jobs',
    ['job_type']
)

# Decorator for automatic metric collection
def track_request_metrics(endpoint: str):
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            method = kwargs.get('request', {}).method
            active_requests.labels(endpoint=endpoint).inc()
            
            start_time = time.time()
            status_code = 200
            
            try:
                result = await func(*args, **kwargs)
                return result
            except HTTPException as e:
                status_code = e.status_code
                raise
            except Exception:
                status_code = 500
                raise
            finally:
                duration = time.time() - start_time
                
                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status_code=status_code
                ).inc()
                
                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)
                
                active_requests.labels(endpoint=endpoint).dec()
        
        return wrapper
    return decorator

# Custom business metrics
class BusinessMetrics:
    @staticmethod
    def record_search_performed(strategy: str, query_complexity: str, duration: float):
        search_queries_total.labels(
            strategy=strategy,
            user_type='api'  # or 'dashboard', 'internal'
        ).inc()
        
        retrieval_latency_seconds.labels(
            strategy=strategy,
            query_complexity=query_complexity
        ).observe(duration)
    
    @staticmethod
    def record_document_processed(file_type: str, status: str, chunks_created: int):
        documents_processed_total.labels(
            file_type=file_type,
            status=status,
            processing_strategy='standard'
        ).inc()
        
        if status == 'success':
            document_chunks_created.observe(chunks_created)
```

### Metrics Endpoint
```python
# src/presentation/api/metrics.py
from fastapi import APIRouter
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

router = APIRouter(tags=["metrics"])

@router.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    metrics = generate_latest()
    return Response(
        content=metrics,
        media_type=CONTENT_TYPE_LATEST
    )
```

## 📝 Structured Logging

### Logging Configuration
```python
# src/infrastructure/monitoring/logging.py
import structlog
import logging
import sys
from pythonjsonlogger import jsonlogger
from src.shared.config.settings import settings

def setup_logging():
    """Configure structured logging for the application"""
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.monitoring.log_level.upper())
    )
    
    # JSON formatter for production
    if settings.environment == "production":
        logHandler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter(
            fmt='%(timestamp)s %(level)s %(name)s %(message)s',
            rename_fields={'levelname': 'level', 'name': 'logger'}
        )
        logHandler.setFormatter(formatter)
        logging.root.handlers = [logHandler]
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ),
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer() if settings.environment == "production" 
            else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

# Logger instance
logger = structlog.get_logger()

# Logging middleware
class LoggingMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        
        # Bind request context
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_host=request.client.host if request.client else None
        )
        
        logger.info("request_started")
        
        start_time = time.time()
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            logger.info(
                "request_completed",
                status_code=response.status_code,
                duration_ms=duration * 1000
            )
            
            response.headers["X-Request-ID"] = request_id
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "request_failed",
                error=str(e),
                duration_ms=duration * 1000,
                exc_info=True
            )
            raise

# Domain event logging
class EventLogger:
    @staticmethod
    def log_document_processed(event: DocumentProcessedEvent):
        logger.info(
            "document_processed",
            document_id=str(event.document_id),
            chunks_created=event.chunks_created,
            processing_time_ms=event.processing_time_ms,
            embedding_model=event.embedding_model
        )
    
    @staticmethod
    def log_search_performed(query: Query, results: RetrievalResult):
        logger.info(
            "search_performed",
            query_id=str(query.id),
            query_text=query.text[:100],  # Truncate for privacy
            strategy=query.strategy.value,
            results_count=len(results.documents),
            avg_score=results.average_score,
            total_time_ms=results.total_time_ms
        )
```

### Log Aggregation with ELK
```yaml
# k8s/logging/fluentd-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
  namespace: kube-system
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/*rag-system*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      read_from_head true
      <parse>
        @type json
        time_format %Y-%m-%dT%H:%M:%S.%NZ
      </parse>
    </source>
    
    <filter kubernetes.**>
      @type kubernetes_metadata
      @id filter_kube_metadata
    </filter>
    
    <filter kubernetes.**>
      @type parser
      key_name log
      reserve_data true
      remove_key_name_field true
      <parse>
        @type json
      </parse>
    </filter>
    
    <match **>
      @type elasticsearch
      host elasticsearch.elastic-system.svc.cluster.local
      port 9200
      logstash_format true
      logstash_prefix rag-system
      include_tag_key true
      type_name _doc
      tag_key @log_name
      <buffer>
        @type memory
        flush_interval 10s
        chunk_limit_size 2M
        queue_limit_length 32
        retry_max_interval 30
      </buffer>
    </match>
```

## 🔍 Distributed Tracing

### OpenTelemetry Setup
```python
# src/infrastructure/monitoring/tracing.py
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

def setup_tracing(app, service_name: str, otlp_endpoint: str):
    """Configure OpenTelemetry tracing"""
    
    # Create resource
    resource = Resource(attributes={
        "service.name": service_name,
        "service.version": settings.version,
        "deployment.environment": settings.environment
    })
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    
    # Configure OTLP exporter
    otlp_exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        insecure=True
    )
    
    # Add batch processor
    span_processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(span_processor)
    
    # Auto-instrumentation
    FastAPIInstrumentor.instrument_app(app)
    SQLAlchemyInstrumentor().instrument(engine=engine)
    RedisInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()
    
    return trace.get_tracer(service_name)

# Custom trace decorators
tracer = trace.get_tracer(__name__)

def trace_method(span_name: str = None):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            name = span_name or f"{func.__module__}.{func.__name__}"
            
            with tracer.start_as_current_span(name) as span:
                # Add common attributes
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                try:
                    # Execute function
                    result = await func(*args, **kwargs)
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result
                    
                except Exception as e:
                    # Record exception
                    span.record_exception(e)
                    span.set_status(
                        trace.Status(trace.StatusCode.ERROR, str(e))
                    )
                    raise
        
        return wrapper
    return decorator

# Trace business operations
class TracedOperations:
    @trace_method("rag.retrieval.search")
    async def traced_search(self, query: Query) -> RetrievalResult:
        span = trace.get_current_span()
        
        # Add query attributes
        span.set_attribute("query.text", query.text[:100])
        span.set_attribute("query.strategy", query.strategy.value)
        span.set_attribute("query.max_results", query.max_results)
        
        # Trace sub-operations
        with tracer.start_as_current_span("embedding_generation"):
            embedding = await self.embedding_service.embed_text(query.text)
            span.set_attribute("embedding.dimensions", len(embedding.vector))
        
        with tracer.start_as_current_span("vector_search"):
            results = await self.vector_store.search(embedding, query.max_results)
            span.set_attribute("results.count", len(results))
        
        with tracer.start_as_current_span("reranking"):
            reranked = await self.reranker.rerank(query.text, results)
            span.set_attribute("reranked.count", len(reranked))
        
        return RetrievalResult(documents=reranked)
```

## 🚨 Alerting Configuration

### Prometheus Alert Rules
```yaml
# k8s/monitoring/prometheus-rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: rag-system-alerts
  namespace: rag-system
spec:
  groups:
  - name: rag-api-alerts
    interval: 30s
    rules:
    
    # High error rate
    - alert: HighErrorRate
      expr: |
        rate(rag_http_requests_total{status_code=~"5.."}[5m]) > 0.05
      for: 5m
      labels:
        severity: critical
        service: rag-api
      annotations:
        summary: "High error rate detected"
        description: "Error rate is {{ $value | humanizePercentage }} for {{ $labels.endpoint }}"
    
    # High latency
    - alert: HighLatency
      expr: |
        histogram_quantile(0.95, rate(rag_http_request_duration_seconds_bucket[5m])) > 2
      for: 5m
      labels:
        severity: warning
        service: rag-api
      annotations:
        summary: "High API latency"
        description: "95th percentile latency is {{ $value }}s for {{ $labels.endpoint }}"
    
    # Database connection pool exhaustion
    - alert: DatabaseConnectionPoolExhausted
      expr: |
        rag_database_connection_pool_size / rag_database_connection_pool_max > 0.9
      for: 5m
      labels:
        severity: warning
        service: database
      annotations:
        summary: "Database connection pool near exhaustion"
        description: "Connection pool {{ $labels.pool_name }} is {{ $value | humanizePercentage }} full"
    
    # Cache hit rate low
    - alert: LowCacheHitRate
      expr: |
        rate(rag_cache_hits_total[5m]) / 
        (rate(rag_cache_hits_total[5m]) + rate(rag_cache_misses_total[5m])) < 0.5
      for: 10m
      labels:
        severity: warning
        service: cache
      annotations:
        summary: "Low cache hit rate"
        description: "Cache hit rate is {{ $value | humanizePercentage }} for {{ $labels.cache_type }}"
    
    # Retrieval performance degradation
    - alert: RetrievalPerformanceDegradation
      expr: |
        histogram_quantile(0.95, rate(rag_retrieval_latency_seconds_bucket[5m])) > 1
      for: 10m
      labels:
        severity: warning
        service: retrieval
      annotations:
        summary: "Retrieval performance degradation"
        description: "95th percentile retrieval latency is {{ $value }}s for strategy {{ $labels.strategy }}"
    
    # Background job queue growing
    - alert: BackgroundJobQueueGrowing
      expr: |
        rag_background_jobs_queued > 1000
      for: 15m
      labels:
        severity: warning
        service: workers
      annotations:
        summary: "Background job queue growing"
        description: "{{ $value }} jobs queued for {{ $labels.job_type }}"
    
    # Document processing failures
    - alert: DocumentProcessingFailures
      expr: |
        rate(rag_documents_processed_total{status="error"}[5m]) > 0.1
      for: 10m
      labels:
        severity: critical
        service: document-processor
      annotations:
        summary: "High document processing failure rate"
        description: "Document processing failure rate is {{ $value | humanizePercentage }}"
```

### AlertManager Configuration
```yaml
# k8s/monitoring/alertmanager-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-config
  namespace: monitoring
data:
  alertmanager.yml: |
    global:
      resolve_timeout: 5m
      slack_api_url: 'YOUR_SLACK_WEBHOOK_URL'
    
    route:
      group_by: ['alertname', 'cluster', 'service']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 12h
      receiver: 'default'
      routes:
      - match:
          severity: critical
        receiver: 'critical-alerts'
        continue: true
      - match:
          severity: warning
        receiver: 'warning-alerts'
    
    receivers:
    - name: 'default'
      slack_configs:
      - channel: '#rag-alerts'
        title: 'RAG System Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ .Annotations.description }}{{ end }}'
    
    - name: 'critical-alerts'
      slack_configs:
      - channel: '#rag-critical'
        title: '🚨 CRITICAL: RAG System Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ .Annotations.description }}{{ end }}'
      pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
    
    - name: 'warning-alerts'
      slack_configs:
      - channel: '#rag-warnings'
        title: '⚠️ Warning: RAG System Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ .Annotations.description }}{{ end }}'
```

## 📊 Dashboards

### Grafana Dashboard Configuration
```json
{
  "dashboard": {
    "title": "RAG System Overview",
    "uid": "rag-overview",
    "panels": [
      {
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "title": "Request Rate",
        "targets": [
          {
            "expr": "sum(rate(rag_http_requests_total[5m])) by (endpoint)",
            "legendFormat": "{{endpoint}}"
          }
        ],
        "type": "graph"
      },
      {
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "title": "Error Rate",
        "targets": [
          {
            "expr": "sum(rate(rag_http_requests_total{status_code=~\"5..\"}[5m])) by (endpoint) / sum(rate(rag_http_requests_total[5m])) by (endpoint)",
            "legendFormat": "{{endpoint}}"
          }
        ],
        "type": "graph",
        "yaxis": {"format": "percent"}
      },
      {
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        "title": "Response Time (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(rag_http_request_duration_seconds_bucket[5m])) by (endpoint, le))",
            "legendFormat": "{{endpoint}}"
          }
        ],
        "type": "graph",
        "yaxis": {"format": "s"}
      },
      {
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
        "title": "Active Requests",
        "targets": [
          {
            "expr": "sum(rag_active_requests) by (endpoint)",
            "legendFormat": "{{endpoint}}"
          }
        ],
        "type": "graph"
      },
      {
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16},
        "title": "Retrieval Strategy Performance",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(rag_retrieval_latency_seconds_bucket[5m])) by (strategy, le))",
            "legendFormat": "{{strategy}}"
          }
        ],
        "type": "graph",
        "yaxis": {"format": "s"}
      },
      {
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 24},
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "sum(rate(rag_cache_hits_total[5m])) by (cache_type) / (sum(rate(rag_cache_hits_total[5m])) by (cache_type) + sum(rate(rag_cache_misses_total[5m])) by (cache_type))",
            "legendFormat": "{{cache_type}}"
          }
        ],
        "type": "graph",
        "yaxis": {"format": "percent"}
      },
      {
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 24},
        "title": "Document Processing Rate",
        "targets": [
          {
            "expr": "sum(rate(rag_documents_processed_total[5m])) by (status)",
            "legendFormat": "{{status}}"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

### Custom Business Dashboard
```python
# src/presentation/api/analytics.py
from fastapi import APIRouter, Depends
from datetime import datetime, timedelta
from typing import Dict, List

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])

@router.get("/dashboard/summary")
async def get_dashboard_summary(
    start_date: datetime = None,
    end_date: datetime = None,
    metrics_service: MetricsService = Depends()
) -> Dict:
    """Get dashboard summary metrics"""
    
    if not start_date:
        start_date = datetime.utcnow() - timedelta(hours=24)
    if not end_date:
        end_date = datetime.utcnow()
    
    return {
        "period": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat()
        },
        "requests": {
            "total": await metrics_service.get_request_count(start_date, end_date),
            "error_rate": await metrics_service.get_error_rate(start_date, end_date),
            "avg_latency_ms": await metrics_service.get_avg_latency(start_date, end_date)
        },
        "retrieval": {
            "total_searches": await metrics_service.get_search_count(start_date, end_date),
            "strategy_breakdown": await metrics_service.get_strategy_breakdown(start_date, end_date),
            "avg_results_per_search": await metrics_service.get_avg_results_per_search(start_date, end_date)
        },
        "documents": {
            "total_processed": await metrics_service.get_documents_processed(start_date, end_date),
            "success_rate": await metrics_service.get_document_success_rate(start_date, end_date),
            "avg_processing_time_s": await metrics_service.get_avg_processing_time(start_date, end_date)
        },
        "system": {
            "active_users": await metrics_service.get_active_users(start_date, end_date),
            "cache_hit_rate": await metrics_service.get_cache_hit_rate(start_date, end_date),
            "database_connections": await metrics_service.get_db_connection_stats()
        }
    }

@router.get("/dashboard/realtime")
async def get_realtime_metrics(
    metrics_service: MetricsService = Depends()
) -> Dict:
    """Get real-time metrics for live dashboard"""
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "active_requests": await metrics_service.get_active_requests(),
        "requests_per_second": await metrics_service.get_current_rps(),
        "error_rate_5m": await metrics_service.get_error_rate_window(minutes=5),
        "avg_latency_1m": await metrics_service.get_avg_latency_window(minutes=1),
        "queue_depth": await metrics_service.get_queue_depth(),
        "system_health": await metrics_service.get_system_health_score()
    }
```

## 🔧 Monitoring Best Practices

### SLI/SLO Definition
```yaml
# slo-config.yaml
service_level_indicators:
  - name: api_availability
    description: "API endpoint availability"
    query: |
      sum(rate(rag_http_requests_total{status_code!~"5.."}[5m])) /
      sum(rate(rag_http_requests_total[5m]))
    
  - name: api_latency
    description: "API response time"
    query: |
      histogram_quantile(0.95, 
        sum(rate(rag_http_request_duration_seconds_bucket[5m])) by (le)
      )
    
  - name: search_quality
    description: "Search result relevance"
    query: |
      avg(rag_search_relevance_score)

service_level_objectives:
  - sli: api_availability
    target: 0.999  # 99.9% availability
    window: 30d
    
  - sli: api_latency
    target: 0.5   # 500ms p95 latency
    window: 30d
    
  - sli: search_quality
    target: 0.85  # 85% relevance score
    window: 7d
```

### Error Budget Monitoring
```python
# src/infrastructure/monitoring/error_budget.py
class ErrorBudgetMonitor:
    def __init__(self, prometheus_client: PrometheusClient):
        self.prometheus = prometheus_client
        self.slos = self._load_slos()
    
    async def calculate_error_budget(self, slo_name: str, window_days: int = 30):
        """Calculate remaining error budget for an SLO"""
        
        slo = self.slos[slo_name]
        
        # Query actual performance
        actual = await self.prometheus.query(
            slo['query'],
            start=datetime.utcnow() - timedelta(days=window_days),
            end=datetime.utcnow()
        )
        
        # Calculate budget
        target = slo['target']
        budget_total = (1 - target) * window_days * 24 * 60  # Minutes
        budget_used = (target - actual) * window_days * 24 * 60
        budget_remaining = budget_total - budget_used
        
        return {
            "slo": slo_name,
            "target": target,
            "actual": actual,
            "budget_total_minutes": budget_total,
            "budget_used_minutes": budget_used,
            "budget_remaining_minutes": budget_remaining,
            "budget_remaining_percentage": (budget_remaining / budget_total) * 100
        }
```

## 📈 Performance Profiling

### Continuous Profiling
```python
# src/infrastructure/monitoring/profiling.py
import cProfile
import pstats
from pyflame import Profiler
import asyncio

class ContinuousProfiler:
    def __init__(self):
        self.profiler = Profiler()
        self.is_profiling = False
    
    async def profile_endpoint(self, func, duration_seconds: int = 60):
        """Profile an endpoint for a specific duration"""
        
        self.is_profiling = True
        profile = cProfile.Profile()
        
        # Start profiling
        profile.enable()
        
        # Run for specified duration
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < duration_seconds:
            if not self.is_profiling:
                break
            await asyncio.sleep(0.1)
        
        # Stop profiling
        profile.disable()
        
        # Generate report
        stats = pstats.Stats(profile)
        stats.sort_stats('cumulative')
        
        return self._generate_flame_graph(stats)
    
    def _generate_flame_graph(self, stats):
        """Generate flame graph from profiling stats"""
        # Implementation for flame graph generation
        pass
```

## 🚀 Monitoring Checklist

### Pre-deployment
- [ ] All metrics endpoints tested
- [ ] Alert rules validated
- [ ] Dashboards imported
- [ ] Log aggregation verified
- [ ] Tracing enabled

### Post-deployment
- [ ] Verify metrics collection
- [ ] Test alert notifications
- [ ] Check dashboard data
- [ ] Validate log parsing
- [ ] Confirm trace sampling

### Regular Reviews
- [ ] Weekly SLO review
- [ ] Monthly alert tuning
- [ ] Quarterly dashboard updates
- [ ] Annual monitoring strategy review