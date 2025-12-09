#!/usr/bin/env python3
"""
Download dati SDSS per analisi weak lensing
Scarica cataloghi di galassie lens e source per galaxy-galaxy lensing
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm
import numpy as np

def download_file(url, dest, desc="Download"):
    """Download file con progress bar"""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    if dest.exists():
        print(f"✓ File già esistente: {dest.name}")
        return True
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=desc
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✅ Scaricato: {dest.name}")
        return True
        
    except Exception as e:
        print(f"❌ Errore download {url}: {e}")
        if dest.exists():
            dest.unlink()
        return False

def download_sdss_catalogs(data_dir='data/sdss'):
    """
    Scarica cataloghi SDSS per lensing
    
    Opzioni:
    1. SDSS DR16 Value-Added Catalogs (raccomandato)
    2. BOSS/eBOSS cataloghi pubblici
    3. Legacy Surveys (alternativa)
    """
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("📥 DOWNLOAD DATI SDSS")
    print("="*60)
    
    # Opzione 1: SDSS DR16 - Galaxy Property Catalog
    # Contiene masse stellari, fotometria, redshift
    print("\n📦 Download catalogo galassie SDSS DR16...")
    
    catalogs = {
        'galaxy_properties': {
            'url': 'https://data.sdss.org/sas/dr16/eboss/lss/catalogs/dr16_QSO.fits',
            'desc': 'Galaxy catalog DR16'
        },
        # Alternativa più leggera per test
        'test_sample': {
            'url': 'https://data.sdss.org/sas/dr12/boss/lss/galaxy_DR12v5_CMASS_North.fits.gz',
            'desc': 'Test sample BOSS'
        }
    }
    
    print("\nATTENZIONE: I file completi sono molto grandi (>10 GB)")
    print("Opzioni:")
    print("  1. Download campione test (~500 MB) - rapido")
    print("  2. Download catalogo completo (~15 GB) - completo")
    print("  3. Genera dati mock per sviluppo - istantaneo")
    
    choice = input("\nScegli opzione (1/2/3): ").strip()
    
    if choice == '1':
        print("\n📥 Download campione test...")
        success = download_file(
            catalogs['test_sample']['url'],
            data_dir / 'boss_test_sample.fits.gz',
            'BOSS test sample'
        )
        if success:
            print("\n✅ Campione test scaricato")
            print("   File: data/sdss/boss_test_sample.fits.gz")
            return True
    
    elif choice == '2':
        print("\n⚠️  ATTENZIONE: Download completo richiede tempo e spazio")
        confirm = input("Confermi? (yes/no): ")
        if confirm.lower() == 'yes':
            print("\n📥 Download catalogo completo...")
            print("   Questo richiederà 30-60 minuti...")
            # TODO: implementare download catalogo completo
            # Richiede accesso API SDSS più sofisticato
            print("\n⚠️  Implementazione download completo in sviluppo")
            print("   Per ora usa opzione 1 (test) o 3 (mock)")
            return False
        else:
            print("Download annullato")
            return False
    
    elif choice == '3':
        print("\n🎲 Generazione dati mock...")
        generate_mock_data(data_dir)
        return True
    
    else:
        print("Opzione non valida")
        return False

def generate_mock_data(data_dir):
    """
    Genera dati mock realistici per sviluppo/test
    Simula catalogo SDSS con proprietà statistiche corrette
    """
    from astropy.table import Table
    from astropy import units as u
    import astropy.coordinates as coord
    
    print("\n🎲 Generazione catalogo mock...")
    
    # Parametri mock
    n_lens = 50000  # Galassie lens
    n_source = 200000  # Galassie source
    
    # Genera galassie lens (quelle di cui misuriamo la massa)
    print(f"   Generazione {n_lens} lens...")
    
    # Masse stellari: distribuzione realistica
    log_mass = np.random.normal(10.5, 0.7, n_lens)  # log10(M*/Msun)
    mass_stellar = 10**log_mass
    
    # Redshift: z ~ 0.2-0.6 (BOSS-like)
    z_lens = np.random.uniform(0.2, 0.6, n_lens)
    
    # Posizioni (RA, Dec) - campione casuale su 1000 deg²
    ra_lens = np.random.uniform(130, 230, n_lens)
    dec_lens = np.random.uniform(0, 50, n_lens)
    
    # Magnitudini
    r_mag = np.random.normal(19.5, 1.0, n_lens)
    
    lens_catalog = Table({
        'ra': ra_lens,
        'dec': dec_lens,
        'z': z_lens,
        'log_mass': log_mass,
        'mass_stellar': mass_stellar,
        'r_mag': r_mag
    })
    
    # Genera galassie source (quelle di cui misuriamo lo shear)
    print(f"   Generazione {n_source} source...")
    
    ra_source = np.random.uniform(130, 230, n_source)
    dec_source = np.random.uniform(0, 50, n_source)
    z_source = np.random.uniform(0.5, 1.2, n_source)  # Dietro le lens
    
    # Shear: componenti tangenziale e cross
    # Per ora random, poi calcoleremo quello reale
    gamma_t = np.random.normal(0, 0.01, n_source)
    gamma_x = np.random.normal(0, 0.01, n_source)
    
    # Shape noise (intrinseco)
    e1 = np.random.normal(0, 0.3, n_source)
    e2 = np.random.normal(0, 0.3, n_source)
    
    source_catalog = Table({
        'ra': ra_source,
        'dec': dec_source,
        'z': z_source,
        'gamma_t': gamma_t,
        'gamma_x': gamma_x,
        'e1': e1,
        'e2': e2
    })
    
    # Salva cataloghi
    lens_file = data_dir / 'mock_lens_catalog.fits'
    source_file = data_dir / 'mock_source_catalog.fits'
    
    lens_catalog.write(lens_file, overwrite=True)
    source_catalog.write(source_file, overwrite=True)
    
    print(f"\n✅ Cataloghi mock generati:")
    print(f"   Lens:   {lens_file} ({len(lens_catalog)} galassie)")
    print(f"   Source: {source_file} ({len(source_catalog)} galassie)")
    
    # Statistiche
    print(f"\n📊 Statistiche mock:")
    print(f"   Massa stellare media: {10**log_mass.mean():.2e} M☉")
    print(f"   Range masse: {10**log_mass.min():.2e} - {10**log_mass.max():.2e} M☉")
    print(f"   Redshift medio lens: {z_lens.mean():.2f}")
    print(f"   Redshift medio source: {z_source.mean():.2f}")
    
    return True

def download_shear_catalogs(data_dir='data/sdss'):
    """
    Download cataloghi shear pre-calcolati
    Alternative: SDSS, KiDS, DES
    """
    print("\n📥 Download cataloghi shear...")
    print("   Shear pre-calcolati non sempre disponibili pubblicamente")
    print("   Useremo dati mock o calcoleremo shear da immagini")
    return True

def main():
    """Main download"""
    print("="*60)
    print("🌌 DOWNLOAD DATI SDSS PER TEST LENSING")
    print("="*60)
    
    # Directory dati
    data_dir = Path(__file__).parent / 'data' / 'sdss'
    
    # Download cataloghi
    success = download_sdss_catalogs(data_dir)
    
    if success:
        print("\n" + "="*60)
        print("✅ DOWNLOAD COMPLETATO")
        print("="*60)
        print("\nProssimo passo:")
        print("  python process_lensing.py  # Elabora dati per test")
        return 0
    else:
        print("\n❌ Download fallito")
        return 1

if __name__ == '__main__':
    sys.exit(main())
