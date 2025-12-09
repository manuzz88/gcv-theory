#!/usr/bin/env python3
"""
Test GPU Setup for GCV Analysis
Verifica che JAX veda le GPU e può fare calcoli
"""

import sys

print("="*60)
print("GCV GPU SETUP TEST")
print("="*60)

# Test 1: Import JAX
print("\n[1/5] Testing JAX import...")
try:
    import jax
    import jax.numpy as jnp
    print("✅ JAX imported successfully")
except ImportError as e:
    print(f"❌ JAX import failed: {e}")
    sys.exit(1)

# Test 2: Check devices
print("\n[2/5] Checking available devices...")
devices = jax.devices()
print(f"Found {len(devices)} device(s):")
for i, dev in enumerate(devices):
    print(f"  Device {i}: {dev}")

gpu_count = sum(1 for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower())
print(f"\n✅ Found {gpu_count} GPU(s)")

if gpu_count == 0:
    print("⚠️  WARNING: No GPU found! Will use CPU (slow)")
elif gpu_count == 2:
    print("🎉 PERFECT: 2 GPUs detected (RTX 4000 Ada)")

# Test 3: Simple GPU computation
print("\n[3/5] Testing GPU computation...")
try:
    # Matrix multiplication on GPU
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1000, 1000))
    y = jnp.dot(x, x.T)
    result = jnp.sum(y)
    print(f"✅ GPU computation successful: result = {result:.4f}")
except Exception as e:
    print(f"❌ GPU computation failed: {e}")
    sys.exit(1)

# Test 4: Check PyMC
print("\n[4/5] Testing PyMC import...")
try:
    import pymc as pm
    print(f"✅ PyMC {pm.__version__} imported")
except ImportError:
    print("⚠️  PyMC not installed. Install with: pip install pymc")

# Test 5: Check other dependencies
print("\n[5/5] Checking other dependencies...")
deps = {
    'numpy': 'numpy',
    'scipy': 'scipy', 
    'matplotlib': 'matplotlib',
    'arviz': 'arviz',
    'corner': 'corner',
    'astropy': 'astropy',
}

missing = []
for name, module in deps.items():
    try:
        __import__(module)
        print(f"  ✅ {name}")
    except ImportError:
        print(f"  ❌ {name} (missing)")
        missing.append(name)

if missing:
    print(f"\n⚠️  Missing: {', '.join(missing)}")
    print("Install with: pip install -r requirements.txt")
else:
    print("\n✅ All dependencies OK!")

# Summary
print("\n" + "="*60)
print("SETUP SUMMARY")
print("="*60)
print(f"GPU Count: {gpu_count}")
print(f"JAX Version: {jax.__version__}")
print(f"Devices: {[str(d) for d in devices]}")

if gpu_count >= 1:
    print("\n🚀 READY TO START GPU TESTS!")
else:
    print("\n⚠️  No GPU - tests will be SLOW")

print("="*60)
