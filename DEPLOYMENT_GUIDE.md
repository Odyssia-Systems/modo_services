# Deployment Guide for Modo Embedding Service

## 🚨 Memory Issue Analysis

**Current Memory Usage: 720.6 MB** (exceeds 512MB Render free tier)

### Memory Breakdown:
- PyTorch + Transformers: ~400MB
- CLIP Model weights: ~200MB  
- FAISS Index: ~12MB
- Python overhead: ~100MB
- **Total: ~720MB** ❌

## 🎯 Deployment Options

### Option 1: Upgrade Render Plan (Recommended)
- **Cost**: ~$7-25/month
- **Memory**: 1-2GB (sufficient)
- **Effort**: Zero code changes
- **Files**: Use existing `main.py`, `embedding_service.py`, `render.yaml`

### Option 2: Lightweight Service (Free Tier)
- **Cost**: Free
- **Memory**: <200MB (fits 512MB limit)
- **Limitation**: No live model inference (mock responses)
- **Files**: Use `main_light.py`, `embedding_service_light.py`, `render_light.yaml`

### Option 3: ONNX Optimization (Advanced)
- **Cost**: Free
- **Memory**: ~300MB (fits 512MB limit)
- **Effort**: Significant refactoring required
- **Benefit**: Full functionality with reduced memory

## 🚀 Quick Deployment Steps

### For Option 1 (Upgraded Plan):
```bash
# 1. Use existing files
git add .
git commit -m "production ready"
git push

# 2. Deploy to Render with render.yaml
# 3. Set environment variables in Render dashboard:
#    - TOKENIZERS_PARALLELISM=false
#    - OMP_NUM_THREADS=1
#    - MKL_NUM_THREADS=1
#    - HF_HOME=/opt/render/project/.hf-cache
#    - TRANSFORMERS_CACHE=/opt/render/project/.hf-cache
```

### For Option 2 (Lightweight):
```bash
# 1. Switch to lightweight files
cp main_light.py main.py
cp embedding_service_light.py embedding_service.py
cp requirements_light.txt requirements.txt
cp render_light.yaml render.yaml

# 2. Deploy
git add .
git commit -m "lightweight service for 512MB limit"
git push
```

## 🧪 Pre-Deployment Testing

Run the production test locally:
```bash
python test_production.py
```

**Expected Results:**
- ✅ Service initialization: ~335MB
- ✅ Model loading: ~720MB (will fail on 512MB limit)
- ✅ Server startup: Working
- ⚠️ Memory warning: Upgrade plan recommended

## 📊 Memory Optimization Tips

### For 512MB Limit:
1. **Remove heavy dependencies**:
   - Remove `sentence-transformers`
   - Remove `torch`
   - Use pre-computed embeddings

2. **Environment variables**:
   ```bash
   TOKENIZERS_PARALLELISM=false
   OMP_NUM_THREADS=1
   MKL_NUM_THREADS=1
   PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
   ```

3. **Server settings**:
   - Single worker: `--workers 1`
   - No reload in production
   - Garbage collection after requests

## 🔧 Render Configuration

### render.yaml (Full Service):
```yaml
services:
  - type: web
    name: modo-embedding-service
    env: python
    plan: starter  # Upgrade to Standard ($7/month) for 1GB
    buildCommand: |
      pip install -r requirements.txt
    startCommand: |
      uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
    envVars:
      - key: TOKENIZERS_PARALLELISM
        value: "false"
      - key: OMP_NUM_THREADS
        value: "1"
      - key: MKL_NUM_THREADS
        value: "1"
```

### render_light.yaml (Lightweight):
```yaml
services:
  - type: web
    name: modo-embedding-service-light
    env: python
    plan: starter  # Free tier
    buildCommand: |
      pip install -r requirements_light.txt
    startCommand: |
      uvicorn main_light:app --host 0.0.0.0 --port $PORT --workers 1
```

## 🎯 Recommendation

**For Production**: Use **Option 1** (upgraded plan)
- Full functionality
- Reliable performance
- Minimal code changes
- Cost: ~$7-25/month

**For Testing**: Use **Option 2** (lightweight)
- Free tier compatible
- Mock responses only
- Good for API structure testing

## 🚨 Troubleshooting

### If deployment fails:
1. Check Render logs for memory errors
2. Verify environment variables are set
3. Ensure single worker configuration
4. Monitor memory usage in Render dashboard

### If service is slow:
1. Enable model caching (already implemented)
2. Use persistent HuggingFace cache
3. Consider CDN for static assets

## 📝 Next Steps

1. **Choose deployment option**
2. **Run production test**: `python test_production.py`
3. **Deploy to Render** with chosen configuration
4. **Monitor memory usage** in Render dashboard
5. **Test endpoints** after deployment
