#!/usr/bin/env python3
"""
Test script to simulate Render production environment
"""
import os
import sys
import time
import psutil
import subprocess
from pathlib import Path

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def test_production_startup():
    """Test production startup with Render-like constraints"""
    print("🧪 Testing production startup...")
    
    # Set production environment variables (like Render)
    env = os.environ.copy()
    env.update({
        'PYTHONUNBUFFERED': '1',
        'TOKENIZERS_PARALLELISM': 'false',
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
        'HF_HOME': '/tmp/.hf-cache',
        'TRANSFORMERS_CACHE': '/tmp/.hf-cache'
    })
    
    print(f"📊 Initial memory: {get_memory_usage():.1f} MB")
    
    # Test 1: Import and initialize service
    print("\n1️⃣ Testing service initialization...")
    try:
        from embedding_service import get_embedding_service
        service = get_embedding_service()
        print(f"✅ Service initialized. Memory: {get_memory_usage():.1f} MB")
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return False
    
    # Test 2: Test model loading (first request simulation)
    print("\n2️⃣ Testing model loading...")
    try:
        # Simulate first search request (loads model)
        test_image = "09a1837f-8936-4b30-949c-81d9eca497cb.jpeg"
        if os.path.exists(test_image):
            result = service.search_similar(test_image, top_k=3)
            if result["success"]:
                print(f"✅ Model loaded successfully. Memory: {get_memory_usage():.1f} MB")
                print(f"   Found {result['total']} results")
            else:
                print(f"❌ Search failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print("⚠️  Test image not found, skipping search test")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # Test 3: Memory usage check
    memory_usage = get_memory_usage()
    print(f"\n📊 Final memory usage: {memory_usage:.1f} MB")
    
    if memory_usage > 400:  # Leave some headroom for 512MB limit
        print(f"⚠️  WARNING: Memory usage ({memory_usage:.1f} MB) is high for 512MB limit")
        print("   Consider upgrading Render plan or using ONNX optimization")
    else:
        print("✅ Memory usage looks good for 512MB limit")
    
    return True

def test_server_startup():
    """Test server startup with production settings"""
    print("\n🚀 Testing server startup...")
    
    # Start server in background
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "main:app", 
        "--host", "127.0.0.1", 
        "--port", "8001",  # Different port
        "--workers", "1"
    ]
    
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            env=os.environ.copy()
        )
        
        # Wait for startup
        time.sleep(5)
        
        if process.poll() is None:
            print("✅ Server started successfully")
            
            # Test health endpoint
            import requests
            try:
                response = requests.get("http://127.0.0.1:8001/health", timeout=10)
                if response.status_code == 200:
                    print("✅ Health endpoint working")
                    print(f"   Response: {response.json()}")
                else:
                    print(f"❌ Health endpoint failed: {response.status_code}")
            except Exception as e:
                print(f"❌ Health check failed: {e}")
            
            # Cleanup
            process.terminate()
            process.wait(timeout=5)
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Server failed to start")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ Server startup test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Production Readiness Test")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("main.py"):
        print("❌ Not in project directory. Run from modo_services/")
        sys.exit(1)
    
    # Test 1: Service initialization
    success1 = test_production_startup()
    
    # Test 2: Server startup
    success2 = test_server_startup()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("✅ All tests passed! Ready for deployment.")
        print("\n📋 Deployment checklist:")
        print("   • Use render.yaml configuration")
        print("   • Set environment variables in Render dashboard")
        print("   • Monitor memory usage after deployment")
    else:
        print("❌ Some tests failed. Fix issues before deploying.")
        if not success1:
            print("   • Fix service initialization issues")
        if not success2:
            print("   • Fix server startup issues")
