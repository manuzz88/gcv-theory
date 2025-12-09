#!/usr/bin/env python3
"""
Test CuPy GPU Setup
Verifica che CuPy veda le GPU RTX 4000 Ada
"""

import sys

print("="*60)
print("CUPY GPU SETUP TEST")
print("="*60)

# Test CuPy
print("\n[1/3] Testing CuPy import...")
try:
    import cupy as cp
    print(f"✅ CuPy {cp.__version__} imported")
except ImportError as e:
    print(f"❌ CuPy import failed: {e}")
    sys.exit(1)

# Check devices
print("\n[2/3] Checking GPU devices...")
try:
    cuda_available = cp.cuda.is_available()
    if cuda_available:
        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"✅ CUDA available")
        print(f"✅ Found {device_count} GPU(s)")
        
        for i in range(device_count):
            device = cp.cuda.Device(i)
            compute_cap = device.compute_capability
            mem_info = device.mem_info
            total_mem_gb = mem_info[1] / 1024**3
            print(f"\n  GPU {i}:")
            print(f"    Compute Capability: {compute_cap}")
            print(f"    Total Memory: {total_mem_gb:.1f} GB")
    else:
        print("❌ CUDA not available")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ GPU check failed: {e}")
    sys.exit(1)

# Test computation
print("\n[3/3] Testing GPU computation...")
try:
    import time
    
    # Small test (avoids overhead)
    size = 5000
    gpu_arr = cp.random.random((size, size))
    
    start = time.time()
    for _ in range(10):  # Multiple operations to amortize overhead
        result = cp.dot(gpu_arr, gpu_arr)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.time() - start
    
    print(f"✅ GPU computation successful")
    print(f"   10 matrix multiplications ({size}x{size}): {gpu_time:.3f}s")
    
except Exception as e:
    print(f"❌ GPU computation failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("GPU SETUP SUMMARY")
print("="*60)
print(f"CuPy Version: {cp.__version__}")
print(f"GPU Count: {device_count}")
print(f"CUDA Available: {cuda_available}")
print("\n🚀 READY FOR GPU-ACCELERATED TESTS!")
print("="*60)
