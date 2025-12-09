#!/usr/bin/env python3
"""
Genera dati mock RIDOTTI per test veloce (~5 min invece di 2 ore)
"""

import numpy as np
from pathlib import Path

def generate_mock_fast():
    """Genera cataloghi mock piccoli"""
    
    print("🎲 Generazione cataloghi mock VELOCI...")
    
    data_dir = Path(__file__).parent / 'data' / 'sdss'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Parametri RIDOTTI per test veloce
    n_lens = 5000      # era 50000
    n_source = 20000   # era 200000
    
    print(f"   Generazione {n_lens} lens...")
    
    np.random.seed(42)
    
    log_mass = np.random.normal(10.5, 0.7, n_lens)
    mass_stellar = 10**log_mass
    z_lens = np.random.uniform(0.2, 0.6, n_lens)
    ra_lens = np.random.uniform(130, 230, n_lens)
    dec_lens = np.random.uniform(0, 50, n_lens)
    r_mag = np.random.normal(19.5, 1.0, n_lens)
    
    lens_data = {
        'ra': ra_lens,
        'dec': dec_lens,
        'z': z_lens,
        'log_mass': log_mass,
        'mass_stellar': mass_stellar,
        'r_mag': r_mag
    }
    
    print(f"   Generazione {n_source} source...")
    
    ra_source = np.random.uniform(130, 230, n_source)
    dec_source = np.random.uniform(0, 50, n_source)
    z_source = np.random.uniform(0.5, 1.2, n_source)
    gamma_t = np.random.normal(0, 0.01, n_source)
    gamma_x = np.random.normal(0, 0.01, n_source)
    e1 = np.random.normal(0, 0.3, n_source)
    e2 = np.random.normal(0, 0.3, n_source)
    
    source_data = {
        'ra': ra_source,
        'dec': dec_source,
        'z': z_source,
        'gamma_t': gamma_t,
        'gamma_x': gamma_x,
        'e1': e1,
        'e2': e2
    }
    
    # Salva
    np.savez(data_dir / 'mock_lens_catalog.npz', **lens_data)
    np.savez(data_dir / 'mock_source_catalog.npz', **source_data)
    
    print(f"\n✅ Cataloghi mock VELOCI generati:")
    print(f"   Lens:   {n_lens} galassie (10x più piccolo)")
    print(f"   Source: {n_source} galassie (10x più piccolo)")
    print(f"   Tempo stimato Test 2: ~5-10 minuti")
    
    return True

if __name__ == '__main__':
    generate_mock_fast()
