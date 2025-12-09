#!/usr/bin/env python3
"""
Setup ambiente per analisi GCV
Controlla GPU, installa dipendenze, prepara directory
"""

import sys
import subprocess
import os
from pathlib import Path

def check_gpu():
    """Controlla disponibilità e info GPU"""
    print("🔍 Controllo GPU...")
    try:
        import cupy as cp
        n_devices = cp.cuda.runtime.getDeviceCount()
        print(f"✅ {n_devices} GPU CUDA disponibili")
        
        for i in range(n_devices):
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props['name'].decode('utf-8')
            mem_gb = props['totalGlobalMem'] / 1e9
            print(f"   GPU {i}: {name}, {mem_gb:.1f} GB")
        return True
    except Exception as e:
        print(f"⚠️  GPU non disponibile: {e}")
        print("   Continuo con CPU (più lento)")
        return False

def check_storage():
    """Controlla spazio disco disponibile"""
    print("\n💾 Controllo storage...")
    import shutil
    stat = shutil.disk_usage(Path.home())
    free_gb = stat.free / 1e9
    print(f"   Spazio libero: {free_gb:.1f} GB")
    
    if free_gb < 50:
        print("⚠️  Meno di 50 GB liberi - potrebbe non bastare")
        resp = input("   Continuare? (y/n): ")
        if resp.lower() != 'y':
            return False
    else:
        print("✅ Storage sufficiente")
    return True

def create_directories():
    """Crea struttura directory"""
    print("\n📁 Creazione directory...")
    
    dirs = [
        'data/sdss',
        'data/clusters',
        'data/processed',
        'results/lensing',
        'results/clusters',
        'plots',
        'cache'
    ]
    
    base = Path(__file__).parent
    for d in dirs:
        path = base / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {d}")
    
    return True

def install_requirements():
    """Installa dipendenze Python"""
    print("\n📦 Installazione dipendenze...")
    req_file = Path(__file__).parent / 'requirements.txt'
    
    if not req_file.exists():
        print("⚠️  File requirements.txt non trovato")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', str(req_file)
        ])
        print("✅ Dipendenze installate")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore installazione: {e}")
        return False

def test_imports():
    """Test import pacchetti critici"""
    print("\n🧪 Test import...")
    
    packages = [
        ('astropy', 'Astronomia'),
        ('numpy', 'Calcolo'),
        ('scipy', 'Analisi'),
        ('matplotlib', 'Plot'),
    ]
    
    all_ok = True
    for pkg, desc in packages:
        try:
            __import__(pkg)
            print(f"   ✓ {desc} ({pkg})")
        except ImportError:
            print(f"   ✗ {desc} ({pkg}) - MANCANTE")
            all_ok = False
    
    # Test GPU opzionale
    try:
        import cupy
        print(f"   ✓ GPU computing (cupy)")
    except ImportError:
        print(f"   ⚠️  GPU computing (cupy) - MANCANTE (opzionale)")
    
    return all_ok

def create_config():
    """Crea file di configurazione"""
    print("\n⚙️  Creazione config...")
    
    config = """# Configurazione analisi GCV
# Modifica questi parametri secondo necessità

[DATA]
# Directory dati
data_dir = data/
sdss_dir = data/sdss/
cluster_dir = data/clusters/

# URL dataset
sdss_dr = 16  # Data Release SDSS
sdss_catalog_url = https://data.sdss.org/sas/dr16/eboss/

[ANALYSIS]
# Parametri lensing
mass_bins = 8, 9, 10, 11, 12  # log10(M*/Msun)
radii_min_kpc = 30
radii_max_kpc = 1000
n_radii = 20

# Parametri GCV
a0 = 1.72e-10  # m/s²
alpha = 2.0    # Rt = alpha * Rc

[COMPUTING]
# Usa GPU se disponibile
use_gpu = True
n_workers = 8  # Thread CPU paralleli
batch_size = 10000  # Per processing GPU

[OUTPUT]
# Directory risultati
results_dir = results/
plots_dir = plots/
save_intermediate = True
"""
    
    config_file = Path(__file__).parent / 'config.ini'
    with open(config_file, 'w') as f:
        f.write(config)
    
    print(f"   ✓ {config_file}")
    return True

def main():
    """Setup completo"""
    print("="*60)
    print("🚀 SETUP AMBIENTE ANALISI GCV")
    print("="*60)
    
    steps = [
        ("GPU", check_gpu),
        ("Storage", check_storage),
        ("Directory", create_directories),
        ("Config", create_config),
        ("Dipendenze", install_requirements),
        ("Test import", test_imports),
    ]
    
    results = {}
    for name, func in steps:
        try:
            results[name] = func()
        except Exception as e:
            print(f"\n❌ Errore in {name}: {e}")
            results[name] = False
    
    print("\n" + "="*60)
    print("📊 RIEPILOGO SETUP")
    print("="*60)
    
    for name, ok in results.items():
        status = "✅" if ok else "❌"
        print(f"{status} {name}")
    
    if all(results.values()):
        print("\n🎉 Setup completato con successo!")
        print("\nProssimi passi:")
        print("  1. python download_sdss.py     # Download dati lensing")
        print("  2. python download_clusters.py # Download dati cluster")
        print("  3. python test2_lensing.py     # Analisi lensing")
        print("  4. python test3_clusters.py    # Analisi cluster")
        return 0
    else:
        print("\n⚠️  Setup incompleto - verifica gli errori sopra")
        return 1

if __name__ == '__main__':
    sys.exit(main())
