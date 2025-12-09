#!/usr/bin/env python3
"""
Genera dati mock semplificati senza dipendenze pesanti
"""

import numpy as np
import json
from pathlib import Path

def generate_mock_data():
    """Genera cataloghi mock in formato semplice"""
    
    print("🎲 Generazione cataloghi mock...")
    
    data_dir = Path(__file__).parent / 'data' / 'sdss'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Parametri
    n_lens = 50000
    n_source = 200000
    
    print(f"   Generazione {n_lens} lens...")
    
    # Galassie lens
    np.random.seed(42)  # Riproducibilità
    
    log_mass = np.random.normal(10.5, 0.7, n_lens)
    mass_stellar = 10**log_mass
    z_lens = np.random.uniform(0.2, 0.6, n_lens)
    ra_lens = np.random.uniform(130, 230, n_lens)
    dec_lens = np.random.uniform(0, 50, n_lens)
    r_mag = np.random.normal(19.5, 1.0, n_lens)
    
    lens_data = {
        'ra': ra_lens.tolist(),
        'dec': dec_lens.tolist(),
        'z': z_lens.tolist(),
        'log_mass': log_mass.tolist(),
        'mass_stellar': mass_stellar.tolist(),
        'r_mag': r_mag.tolist()
    }
    
    print(f"   Generazione {n_source} source...")
    
    # Galassie source
    ra_source = np.random.uniform(130, 230, n_source)
    dec_source = np.random.uniform(0, 50, n_source)
    z_source = np.random.uniform(0.5, 1.2, n_source)
    gamma_t = np.random.normal(0, 0.01, n_source)
    gamma_x = np.random.normal(0, 0.01, n_source)
    e1 = np.random.normal(0, 0.3, n_source)
    e2 = np.random.normal(0, 0.3, n_source)
    
    source_data = {
        'ra': ra_source.tolist(),
        'dec': dec_source.tolist(),
        'z': z_source.tolist(),
        'gamma_t': gamma_t.tolist(),
        'gamma_x': gamma_x.tolist(),
        'e1': e1.tolist(),
        'e2': e2.tolist()
    }
    
    # Salva come numpy arrays
    np.savez(data_dir / 'mock_lens_catalog.npz', **lens_data)
    np.savez(data_dir / 'mock_source_catalog.npz', **source_data)
    
    print(f"\n✅ Cataloghi mock generati:")
    print(f"   Lens:   {data_dir}/mock_lens_catalog.npz ({n_lens} galassie)")
    print(f"   Source: {data_dir}/mock_source_catalog.npz ({n_source} galassie)")
    
    print(f"\n📊 Statistiche:")
    print(f"   Massa media: {10**log_mass.mean():.2e} M☉")
    print(f"   Range masse: {10**log_mass.min():.2e} - {10**log_mass.max():.2e} M☉")
    print(f"   z medio lens: {z_lens.mean():.2f}")
    print(f"   z medio source: {z_source.mean():.2f}")
    
    return True

if __name__ == '__main__':
    generate_mock_data()
