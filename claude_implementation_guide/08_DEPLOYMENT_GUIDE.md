# Deployment Guide - RAG Comprehensive System

## 🚀 Deployment Overview

### Deployment Environments
1. **Development**: Local Docker Compose
2. **Staging**: Kubernetes on cloud provider
3. **Production**: Multi-region Kubernetes with HA

### Deployment Strategy
- **Blue-Green Deployment** for zero-downtime
- **Canary Releases** for gradual rollout
- **Rollback Capability** within 5 minutes
- **Database Migrations** with backward compatibility

## 🐳 Docker Configuration

### Multi-stage Dockerfile
```dockerfile
# docker/Dockerfile.api
# Stage 1: Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.7.0

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root --only main

# Stage 2: Runtime stage
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src ./src
COPY alembic ./alembic
COPY alembic.ini ./

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.presentation.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose for Development
```yaml
# docker-compose.yml
version: '3.9'

services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/rag_system
      - REDIS_URL=redis://redis:6379
      - ENVIRONMENT=development
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./src:/app/src:ro
      - model-cache:/app/models
    networks:
      - rag-network

  worker:
    build:
      context: .
      dockerfile: docker/Dockerfile.worker
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/rag_system
      - REDIS_URL=redis://redis:6379
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./src:/app/src:ro
      - model-cache:/app/models
    networks:
      - rag-network

  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: rag_system
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - rag-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - rag-network

  neo4j:
    image: neo4j:5.0
    environment:
      NEO4J_AUTH: neo4j/password123
      NEO4J_PLUGINS: '["apoc", "graph-data-science"]'
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j-data:/data
    networks:
      - rag-network

volumes:
  postgres-data:
  redis-data:
  neo4j-data:
  model-cache:

networks:
  rag-network:
    driver: bridge
```

## ☸️ Kubernetes Deployment

### Namespace Configuration
```yaml
# k8s/base/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-system
  labels:
    name: rag-system
    environment: production
```

### ConfigMap
```yaml
# k8s/base/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-config
  namespace: rag-system
data:
  ENVIRONMENT: "production"
  API_V1_PREFIX: "/api/v1"
  LOG_LEVEL: "INFO"
  LOG_FORMAT: "json"
  REDIS_URL: "redis://redis-service:6379"
  VECTOR_STORE_TYPE: "pgvector"
  DEFAULT_RETRIEVAL_STRATEGY: "hybrid"
  MAX_RETRIEVAL_RESULTS: "10"
  SIMILARITY_THRESHOLD: "0.7"
```

### Secrets
```yaml
# k8s/base/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: rag-secrets
  namespace: rag-system
type: Opaque
stringData:
  DATABASE_URL: "postgresql+asyncpg://user:pass@postgres:5432/rag_system"
  OPENAI_API_KEY: "your-openai-key"
  ANTHROPIC_API_KEY: "your-anthropic-key"
  SECRET_KEY: "your-secret-key-for-jwt"
  SENTRY_DSN: "your-sentry-dsn"
```

### Deployment
```yaml
# k8s/base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  namespace: rag-system
  labels:
    app: rag-api
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: rag-api
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: api
        image: your-registry/rag-system:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
        - containerPort: 9090
          name: metrics
          protocol: TCP
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: DATABASE_URL
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: OPENAI_API_KEY
        envFrom:
        - configMapRef:
            name: rag-config
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /api/v1/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: model-cache
          mountPath: /app/models
        - name: tmp
          mountPath: /tmp
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      - name: tmp
        emptyDir: {}
```

### Service
```yaml
# k8s/base/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
  namespace: rag-system
  labels:
    app: rag-api
spec:
  type: ClusterIP
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: 9090
    protocol: TCP
  selector:
    app: rag-api
```

### Ingress
```yaml
# k8s/base/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-api-ingress
  namespace: rag-system
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.rag-system.com
    secretName: rag-api-tls
  rules:
  - host: api.rag-system.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-api-service
            port:
              number: 80
```

### HorizontalPodAutoscaler
```yaml
# k8s/base/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-hpa
  namespace: rag-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

### PersistentVolume for Model Cache
```yaml
# k8s/base/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache-pvc
  namespace: rag-system
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 100Gi
```

## 🎯 Helm Chart

### Chart Structure
```
helm/rag-system/
├── Chart.yaml
├── values.yaml
├── values.production.yaml
├── values.staging.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── secrets.yaml
│   ├── hpa.yaml
│   ├── pdb.yaml
│   └── _helpers.tpl
└── charts/
    ├── postgresql/
    └── redis/
```

### values.yaml
```yaml
# helm/rag-system/values.yaml
replicaCount: 3

image:
  repository: your-registry/rag-system
  pullPolicy: IfNotPresent
  tag: ""

imagePullSecrets: []
nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations: {}
  name: ""

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
  hosts:
    - host: api.rag-system.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: rag-api-tls
      hosts:
        - api.rag-system.com

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

postgresql:
  enabled: true
  auth:
    database: rag_system
    username: rag_user
    existingSecret: postgres-secret
  primary:
    persistence:
      enabled: true
      size: 100Gi
      storageClass: fast-ssd
  metrics:
    enabled: true

redis:
  enabled: true
  auth:
    enabled: true
    existingSecret: redis-secret
  master:
    persistence:
      enabled: true
      size: 10Gi
  metrics:
    enabled: true

config:
  environment: production
  logLevel: INFO
  retrieval:
    defaultStrategy: hybrid
    maxResults: 10
    similarityThreshold: 0.7
  multimodal:
    enabled: true
    visionModel: gpt-4-vision-preview

secrets:
  create: true
  openaiApiKey: ""
  anthropicApiKey: ""
  secretKey: ""
  databaseUrl: ""
```

### Helm Deployment Commands
```bash
# Add dependencies
helm dependency update ./helm/rag-system

# Install/Upgrade
helm upgrade --install rag-system ./helm/rag-system \
  --namespace rag-system \
  --create-namespace \
  -f ./helm/rag-system/values.yaml \
  -f ./helm/rag-system/values.production.yaml \
  --set image.tag=v1.2.3 \
  --wait

# Rollback
helm rollback rag-system 1 --namespace rag-system

# Delete
helm delete rag-system --namespace rag-system
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    tags:
      - 'v*'

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        file: docker/Dockerfile.api
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        build-args: |
          VERSION=${{ github.ref_name }}
          BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
          VCS_REF=${{ github.sha }}

  deploy-staging:
    needs: build-and-push
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Configure kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'v1.28.0'
    
    - name: Set up Kubeconfig
      run: |
        echo "${{ secrets.STAGING_KUBECONFIG }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
    
    - name: Deploy to Staging
      run: |
        helm upgrade --install rag-system ./helm/rag-system \
          --namespace rag-staging \
          --create-namespace \
          -f ./helm/rag-system/values.yaml \
          -f ./helm/rag-system/values.staging.yaml \
          --set image.tag=${{ github.ref_name }} \
          --wait \
          --timeout 10m

  integration-tests:
    needs: deploy-staging
    runs-on: ubuntu-latest
    
    steps:
    - name: Run E2E Tests
      run: |
        npm install -g newman
        newman run tests/e2e/postman_collection.json \
          --environment tests/e2e/staging_environment.json \
          --reporters cli,junit \
          --reporter-junit-export test-results.xml
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      with:
        name: test-results
        path: test-results.xml

  deploy-production:
    needs: [build-and-push, integration-tests]
    runs-on: ubuntu-latest
    environment: production
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Configure kubectl
      uses: azure/setup-kubectl@v3
    
    - name: Set up Kubeconfig
      run: |
        echo "${{ secrets.PRODUCTION_KUBECONFIG }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
    
    - name: Deploy to Production (Canary)
      run: |
        # Deploy canary version (10% traffic)
        helm upgrade --install rag-system-canary ./helm/rag-system \
          --namespace rag-production \
          -f ./helm/rag-system/values.yaml \
          -f ./helm/rag-system/values.production.yaml \
          -f ./helm/rag-system/values.canary.yaml \
          --set image.tag=${{ github.ref_name }} \
          --set canary.enabled=true \
          --set canary.weight=10 \
          --wait
    
    - name: Monitor Canary
      run: |
        # Wait and monitor metrics
        sleep 300
        # Check error rate, latency, etc.
        ./scripts/check_canary_health.sh
    
    - name: Promote to Full Production
      if: success()
      run: |
        helm upgrade --install rag-system ./helm/rag-system \
          --namespace rag-production \
          -f ./helm/rag-system/values.yaml \
          -f ./helm/rag-system/values.production.yaml \
          --set image.tag=${{ github.ref_name }} \
          --set canary.enabled=false \
          --wait
```

## 🔐 Security Hardening

### Network Policies
```yaml
# k8s/base/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: rag-api-network-policy
  namespace: rag-system
spec:
  podSelector:
    matchLabels:
      app: rag-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    - podSelector:
        matchLabels:
          app: rag-api
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to:
    - namespaceSelector: {}
      podSelector:
        matchLabels:
          k8s-app: kube-dns
    ports:
    - protocol: UDP
      port: 53
```

### Pod Security Policy
```yaml
# k8s/base/pod-security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: rag-restricted
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  hostNetwork: false
  hostIPC: false
  hostPID: false
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  supplementalGroups:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
  readOnlyRootFilesystem: true
```

## 🔄 Database Migrations

### Migration Strategy
```bash
#!/bin/bash
# scripts/migrate.sh

# Run migrations in init container
kubectl run migration-job \
  --image=your-registry/rag-system:latest \
  --rm \
  --restart=Never \
  --namespace=rag-system \
  --env="DATABASE_URL=$DATABASE_URL" \
  --command -- alembic upgrade head

# Verify migration
kubectl exec -it deployment/rag-api -- alembic current
```

### Rollback Plan
```bash
#!/bin/bash
# scripts/rollback.sh

# Get current revision
CURRENT_REV=$(kubectl exec deployment/rag-api -- alembic current | grep -oE '[a-f0-9]{12}')

# Rollback to previous revision
kubectl exec deployment/rag-api -- alembic downgrade -1

# Rollback application
helm rollback rag-system --namespace rag-system
```

## 📊 Monitoring Setup

### Prometheus ServiceMonitor
```yaml
# k8s/monitoring/service-monitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: rag-api-metrics
  namespace: rag-system
spec:
  selector:
    matchLabels:
      app: rag-api
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "RAG System Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(rag_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rag_request_duration_seconds_bucket)"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(rag_requests_total{status=\"error\"}[5m])"
          }
        ]
      }
    ]
  }
}
```

## 🚨 Disaster Recovery

### Backup Strategy
```bash
#!/bin/bash
# scripts/backup.sh

# Backup PostgreSQL
kubectl exec -it statefulset/postgres -- \
  pg_dump -U postgres rag_system | \
  gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz

# Backup Redis
kubectl exec -it deployment/redis -- \
  redis-cli --rdb /tmp/dump.rdb

# Upload to S3
aws s3 cp backup_*.sql.gz s3://rag-backups/postgres/
aws s3 cp dump.rdb s3://rag-backups/redis/
```

### Recovery Procedure
1. **Restore Database**
   ```bash
   kubectl exec -i statefulset/postgres -- \
     psql -U postgres rag_system < backup.sql
   ```

2. **Restore Redis**
   ```bash
   kubectl cp dump.rdb redis-pod:/data/dump.rdb
   kubectl exec redis-pod -- redis-cli shutdown
   ```

3. **Verify Application**
   ```bash
   kubectl rollout status deployment/rag-api
   kubectl exec deployment/rag-api -- python -m scripts.health_check
   ```

## 📝 Deployment Checklist

### Pre-deployment
- [ ] All tests passing
- [ ] Security scan completed
- [ ] Performance benchmarks acceptable
- [ ] Documentation updated
- [ ] Migration scripts tested
- [ ] Rollback plan documented

### Deployment
- [ ] Database backup taken
- [ ] Monitoring alerts configured
- [ ] Deploy to staging first
- [ ] Run smoke tests
- [ ] Deploy canary (10%)
- [ ] Monitor metrics for 15 mins
- [ ] Full production rollout
- [ ] Verify all services healthy

### Post-deployment
- [ ] Monitor error rates
- [ ] Check performance metrics
- [ ] Verify log aggregation
- [ ] Update status page
- [ ] Notify stakeholders
- [ ] Document any issues