# 🚀 Climate Analytics Platform - Deployment Guide

## Quick Start (5 Minutes)

### Option 1: Local Development
```bash
# 1. Clone/Create project structure
mkdir climate-analytics-platform && cd climate-analytics-platform

# 2. Backend setup
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
echo "DATABASE_URL=sqlite:///./climate_analytics.db" > .env
echo "SECRET_KEY=dev-secret-key" >> .env
echo "DEBUG=True" >> .env

# 3. Frontend setup (new terminal)
cd frontend
npm install
echo "REACT_APP_API_URL=http://localhost:8000" > .env

# 4. Start services
# Terminal 1: Backend
cd backend && python -m uvicorn app.main:app --reload

# Terminal 2: Frontend  
cd frontend && npm start

# 5. Access application
# Frontend: http://localhost:3000
# API: http://localhost:8000/docs
```

### Option 2: Docker (Recommended for Production)
```bash
# 1. Clone project
git clone <repo-url>
cd climate-analytics-platform

# 2. Start with Docker
docker-compose up --build

# 3. Access application
# Frontend: http://localhost:3000
# API: http://localhost:8000
```

## 🌐 Production Deployment Options

### 1. Vercel + Railway (Recommended)
**Frontend (Vercel):**
```bash
# Deploy frontend to Vercel
npm install -g vercel
cd frontend
vercel --prod
```

**Backend (Railway):**
```bash
# Deploy backend to Railway
npm install -g @railway/cli
cd backend
railway login
railway deploy
```

### 2. Docker + VPS
```bash
# Production docker-compose
docker-compose -f docker-compose.prod.yml up -d

# With SSL/HTTPS
docker-compose -f docker-compose.ssl.yml up -d
```

### 3. Kubernetes
```bash
# Apply K8s manifests
kubectl apply -f k8s/
kubectl get pods -l app=climate-analytics
```

## 🔧 Environment Configuration

### Backend (.env)
```env
# Database
DATABASE_URL=sqlite:///./climate_analytics.db  # or PostgreSQL URL

# Security
SECRET_KEY=your-super-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here

# Application
DEBUG=False
ENVIRONMENT=production
LOG_LEVEL=INFO

# CORS
ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# File Upload
MAX_UPLOAD_SIZE=10485760  # 10MB
UPLOAD_DIRECTORY=data/uploads

# ML Models
MODEL_DIRECTORY=data/models
SENTIMENT_CONFIDENCE_THRESHOLD=0.6

# External APIs (optional)
OPENAI_API_KEY=your-openai-key  # for advanced NLP
HUGGINGFACE_API_TOKEN=your-hf-token  # for transformer models
```

### Frontend (.env)
```env
# API Configuration
REACT_APP_API_URL=https://your-backend-url.com
REACT_APP_API_TIMEOUT=30000

# Application
REACT_APP_ENVIRONMENT=production
REACT_APP_VERSION=1.0.0

# Analytics (optional)
REACT_APP_GA_TRACKING_ID=GA-XXXXXXXXX
REACT_APP_HOTJAR_ID=your-hotjar-id

# Feature Flags
REACT_APP_ENABLE_UPLOAD=true
REACT_APP_ENABLE_REPORTS=true
REACT_APP_ENABLE_BATCH_ANALYSIS=true
```

## 📊 Performance Optimization

### Backend Optimizations
```python
# In production, use these settings:

# app/config/settings.py
WORKERS = 4  # CPU cores * 2
WORKER_CLASS = "uvicorn.workers.UvicornWorker"
MAX_CONNECTIONS = 1000
KEEPALIVE = 2

# Use Redis for caching
REDIS_URL = "redis://localhost:6379"
CACHE_TTL = 3600

# Database optimization
DATABASE_POOL_SIZE = 20
DATABASE_MAX_OVERFLOW = 30
```

### Frontend Optimizations
```json
// package.json build optimization
{
  "scripts": {
    "build": "GENERATE_SOURCEMAP=false react-scripts build",
    "build:analyze": "npm run build && npx bundle-analyzer build/static/js/*.js"
  }
}
```

## 🔒 Security Configuration

### 1. API Security
```python
# Enable in production
CORS_CREDENTIALS = True
SECURE_COOKIES = True
CSRF_PROTECTION = True
RATE_LIMITING = True

# Headers
SECURE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block"
}
```

### 2. Database Security
```bash
# PostgreSQL production setup
CREATE USER climate_user WITH PASSWORD 'secure_password';
CREATE DATABASE climate_analytics OWNER climate_user;
GRANT ALL PRIVILEGES ON DATABASE climate_analytics TO climate_user;

# Connection string
DATABASE_URL=postgresql://climate_user:secure_password@localhost/climate_analytics
```

### 3. SSL/HTTPS Setup
```nginx
# nginx SSL configuration
server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    ssl_certificate /etc/ssl/certs/yourdomain.crt;
    ssl_certificate_key /etc/ssl/private/yourdomain.key;
    
    location / {
        proxy_pass http://frontend:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /api/ {
        proxy_pass http://backend:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📈 Monitoring & Logging

### 1. Application Monitoring
```python
# Add to requirements.txt
prometheus-client==0.16.0
structlog==23.1.0

# In main.py
from prometheus_client import Counter, Histogram, generate_latest
import structlog

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_DURATION.observe(duration)
    
    return response
```

### 2. Error Tracking
```bash
# Add Sentry for error tracking
pip install sentry-sdk[fastapi]

# In main.py
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[FastApiIntegration(auto_enabling=True)],
    traces_sample_rate=0.1,
)
```

### 3. Log Management
```yaml
# docker-compose.yml logging
services:
  backend:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## 🧪 Testing & CI/CD

### 1. Automated Testing
```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          cd backend
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: |
          cd backend
          pytest tests/ --cov=app --cov-report=xml
      
      - name: Frontend tests
        run: |
          cd frontend
          npm install
          npm test -- --coverage --watchAll=false
```

### 2. Deployment Pipeline
```yaml
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to Railway
        uses: railway-app/railway-deploy@v1
        with:
          api_token: ${{ secrets.RAILWAY_TOKEN }}
          project_id: ${{ secrets.RAILWAY_PROJECT_ID }}
      
      - name: Deploy to Vercel
        uses: vercel/action@v1
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
```

## 🔧 Troubleshooting

### Common Issues

**1. CORS Errors**
```python
# Fix CORS in main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Update this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**2. Memory Issues**
```python
# Optimize memory usage
import gc
from sklearn.externals import joblib

# Use joblib for model persistence
joblib.dump(model, 'model.pkl', compress=3)

# Clear memory periodically
gc.collect()
```

**3. Slow API Responses**
```python
# Add response caching
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_sentiment_analysis(text_hash):
    return sentiment_analyzer.predict(text)
```

**4. Database Connection Issues**
```python
# Add connection pooling
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True
)
```

## 📋 Production Checklist

### Pre-Deployment
- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Database migrations run
- [ ] Static files collected
- [ ] Security headers configured
- [ ] Rate limiting enabled
- [ ] Monitoring setup
- [ ] Error tracking configured
- [ ] Backup strategy implemented

### Post-Deployment
- [ ] Health checks passing
- [ ] API endpoints responding
- [ ] Frontend loading correctly
- [ ] Authentication working
- [ ] File uploads functioning
- [ ] Database queries optimized
- [ ] Logs being collected
- [ ] Metrics being tracked

## 🚀 Scaling Considerations

### Horizontal Scaling
```yaml
# Kubernetes horizontal pod autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: climate-analytics-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: climate-analytics-backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Database Scaling
```python
# Read/Write splitting
DATABASE_WRITE_URL = "postgresql://user:pass@primary-db/climate"
DATABASE_READ_URL = "postgresql://user:pass@replica-db/climate"

# Use read replicas for analytics queries
```

### CDN Configuration
```javascript
// CloudFront/CDN setup for static assets
const cdnConfig = {
  origins: [{
    domainName: 'your-app.vercel.app',
    customOriginConfig: {
      HTTPPort: 443,
      OriginProtocolPolicy: 'https-only'
    }
  }],
  defaultCacheBehavior: {
    compress: true,
    viewerProtocolPolicy: 'redirect-to-https'
  }
};
```

---

## 🎉 Congratulations!

Your Climate Analytics Platform is now production-ready! 

**Next Steps:**
1. Monitor your deployment
2. Gather user feedback
3. Implement additional features
4. Scale based on usage patterns

**Support:**
- 📧 Email: support@climate-analytics.com
- 📖 Documentation: `/docs`
- 🐛 Issues: GitHub Issues
- 💬 Community: Discord/Slack
