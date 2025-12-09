#!/usr/bin/env python3
"""
Genera dati mock REALISTICI per weak lensing
Con shear tangenziale correlato alla massa della lens
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm

# Costanti fisiche
G = 6.6743e-11  # m³ kg⁻¹ s⁻²
M_sun = 1.9885e30  # kg
kpc = 3.0857e19  # m
c = 2.998e8  # m/s

# Parametri GCV (per generare shear realistico)
A0 = 1.72e-10  # m/s²
ALPHA = 2.0

def angular_separation(ra1, dec1, ra2, dec2):
    """Separazione angolare in gradi (haversine)"""
    ra1, dec1 = np.radians(ra1), np.radians(dec1)
    ra2, dec2 = np.radians(ra2), np.radians(dec2)
    
    dra = ra2 - ra1
    ddec = dec2 - dec1
    
    a = np.sin(ddec/2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return np.degrees(c)

def physical_separation(theta_deg, z_lens):
    """Distanza fisica in kpc"""
    # Distanza angolare [Mpc] - approssimata per z < 1
    D_A = 3000 * z_lens / (1 + z_lens)  # Mpc
    r_phys = D_A * 1000 * np.radians(theta_deg)  # kpc
    return r_phys

def calculate_true_shear(M_lens_Msun, r_kpc, z_lens, z_source):
    """
    Calcola shear tangenziale VERO da un profilo GCV
    
    Parameters
    ----------
    M_lens_Msun : float
        Massa lens [M☉]
    r_kpc : float
        Distanza proiettata [kpc]
    z_lens, z_source : float
        Redshift lens e source
        
    Returns
    -------
    gamma_t : float
        Shear tangenziale
    """
    # Massa in SI
    M_lens = M_lens_Msun * M_sun
    
    # Velocità piatta GCV
    v_inf = (G * M_lens * A0)**(0.25)  # m/s
    
    # Raggio caratteristico
    Rc = np.sqrt(G * M_lens / A0) / kpc  # kpc
    Rt = ALPHA * Rc
    
    # Densità superficiale proiettata (semplificata)
    # Per profilo con transizione
    if r_kpc < Rt:
        # Regime SIS-like
        Sigma = v_inf**2 / (4 * np.pi * G * r_kpc * kpc)
    else:
        # Regime transizione
        Sigma = v_inf**2 / (4 * np.pi * G * Rt * kpc) * (Rt / r_kpc)**1.7
    
    # Critical surface density (approssimato)
    # Σ_crit ~ c² / (4πG) × D_s / (D_l × D_ls)
    # Per z_l ~ 0.4, z_s ~ 0.8 tipico
    D_ratio = z_source / (z_lens * (z_source - z_lens) + 0.1)
    Sigma_crit = (c**2 / (4 * np.pi * G)) * D_ratio / kpc**2
    
    # Shear: γ_t ≈ ΔΣ / Σ_crit
    # ΔΣ ≈ Σ per profili semplici
    gamma_t = Sigma / Sigma_crit
    
    return gamma_t

def generate_realistic_lensing_data():
    """Genera cataloghi con shear REALISTICO"""
    
    print("="*60)
    print("🎲 GENERAZIONE CATALOGHI MOCK REALISTICI")
    print("="*60)
    
    data_dir = Path(__file__).parent / 'data' / 'sdss'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Parametri
    n_lens = 3000      # Ridotto per velocità
    n_source = 15000   # 5x più source che lens
    
    np.random.seed(42)
    
    print(f"\n📊 Parametri:")
    print(f"   Lens: {n_lens}")
    print(f"   Source: {n_source}")
    print(f"   Rapporto: {n_source/n_lens:.1f}:1")
    
    # ========= GALASSIE LENS =========
    print(f"\n📍 Generazione {n_lens} lens...")
    
    log_mass = np.random.normal(10.5, 0.7, n_lens)
    mass_stellar = 10**log_mass
    z_lens = np.random.uniform(0.3, 0.5, n_lens)  # Range più stretto
    
    # Posizioni concentrate (per avere overlap con source)
    ra_lens = np.random.uniform(180, 200, n_lens)
    dec_lens = np.random.uniform(20, 30, n_lens)
    
    r_mag = np.random.normal(19.5, 1.0, n_lens)
    
    lens_data = {
        'ra': ra_lens,
        'dec': dec_lens,
        'z': z_lens,
        'log_mass': log_mass,
        'mass_stellar': mass_stellar,
        'r_mag': r_mag
    }
    
    # ========= GALASSIE SOURCE =========
    print(f"\n🌌 Generazione {n_source} source...")
    
    # Source dietro le lens, stessa regione del cielo
    ra_source = np.random.uniform(180, 200, n_source)
    dec_source = np.random.uniform(20, 30, n_source)
    z_source = np.random.uniform(0.6, 1.0, n_source)  # Dietro le lens
    
    # Shape intrinseco (ellitticità)
    e1_intrinsic = np.random.normal(0, 0.3, n_source)
    e2_intrinsic = np.random.normal(0, 0.3, n_source)
    
    # ========= CALCOLO SHEAR REALISTICO =========
    print(f"\n⚙️  Calcolo shear realistico...")
    print(f"   (questo può richiedere qualche minuto)")
    
    gamma_t = np.zeros(n_source)
    gamma_x = np.zeros(n_source)
    
    # Per ogni source, somma contributo di tutte le lens vicine
    for i_source in tqdm(range(n_source), desc="Shear"):
        
        # Trova lens entro ~20 gradi (per velocità)
        theta_all = angular_separation(
            ra_source[i_source], dec_source[i_source],
            ra_lens, dec_lens
        )
        
        mask_near = (theta_all < 20) & (z_source[i_source] > z_lens)
        lens_near = np.where(mask_near)[0]
        
        # Somma contributi
        for i_lens in lens_near:
            theta = theta_all[i_lens]
            r_phys = physical_separation(theta, z_lens[i_lens])
            
            # Solo se distanza fisica ragionevole (30-1000 kpc)
            if 30 < r_phys < 1000:
                # Calcola shear da questa lens
                shear = calculate_true_shear(
                    mass_stellar[i_lens],
                    r_phys,
                    z_lens[i_lens],
                    z_source[i_source]
                )
                
                # Accumula shear tangenziale
                gamma_t[i_source] += shear
        
        # Aggiungi shape noise
        gamma_t[i_source] += np.random.normal(0, 0.01)
        gamma_x[i_source] = np.random.normal(0, 0.01)
    
    source_data = {
        'ra': ra_source,
        'dec': dec_source,
        'z': z_source,
        'gamma_t': gamma_t,
        'gamma_x': gamma_x,
        'e1': e1_intrinsic,
        'e2': e2_intrinsic
    }
    
    # ========= SALVA =========
    print(f"\n💾 Salvataggio...")
    
    np.savez(data_dir / 'mock_lens_catalog.npz', **lens_data)
    np.savez(data_dir / 'mock_source_catalog.npz', **source_data)
    
    print(f"\n✅ Cataloghi REALISTICI generati:")
    print(f"   Lens:   {n_lens} galassie")
    print(f"   Source: {n_source} galassie")
    
    # ========= STATISTICHE SHEAR =========
    print(f"\n📊 Statistiche shear:")
    print(f"   gamma_t medio: {np.mean(gamma_t):.6f}")
    print(f"   gamma_t std: {np.std(gamma_t):.6f}")
    print(f"   gamma_t range: [{np.min(gamma_t):.6f}, {np.max(gamma_t):.6f}]")
    
    # Conta quante source hanno shear significativo
    n_signal = np.sum(np.abs(gamma_t) > 0.001)
    print(f"   Source con |γ_t| > 0.001: {n_signal}/{n_source} ({100*n_signal/n_source:.1f}%)")
    
    # Statistiche massa
    print(f"\n📊 Statistiche lens:")
    print(f"   Massa media: {10**log_mass.mean():.2e} M☉")
    print(f"   Range masse: {10**log_mass.min():.2e} - {10**log_mass.max():.2e} M☉")
    print(f"   z medio: {z_lens.mean():.2f}")
    
    print(f"\n📊 Statistiche source:")
    print(f"   z medio: {z_source.mean():.2f}")
    print(f"   z range: {z_source.min():.2f} - {z_source.max():.2f}")
    
    print(f"\n✅ DATI REALISTICI PRONTI PER TEST")
    print(f"   Shear correlato con massa lens!")
    print(f"   Tempo stimato Test 2: ~3-5 minuti")
    
    return True

if __name__ == '__main__':
    generate_realistic_lensing_data()
