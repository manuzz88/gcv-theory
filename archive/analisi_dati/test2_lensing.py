#!/usr/bin/env python3
"""
TEST 2: WEAK LENSING GRAVITAZIONALE

Analisi completa:
1. Carica cataloghi lens e source
2. Cross-match e calcolo separazioni
3. Binning per massa barionica
4. Stacking shear → ΔΣ(R)
5. Calcolo predizioni GCV
6. Confronto statistico e verdetto
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
try:
    from tqdm import tqdm
except ImportError:
    # Fallback se tqdm non disponibile
    def tqdm(iterable, **kwargs):
        return iterable

# Costanti fisiche
G = 6.6743e-11  # m³ kg⁻¹ s⁻²
M_sun = 1.9885e30  # kg
kpc = 3.0857e19  # m
c = 2.998e8  # m/s

# Parametri GCV
A0 = 1.72e-10  # m/s²
ALPHA = 2.0  # Rt = alpha * Rc

class LensingAnalysis:
    """Analisi weak lensing per test GCV"""
    
    def __init__(self, lens_file, source_file, use_gpu=True):
        """
        Inizializza analisi
        
        Parameters
        ----------
        lens_file : str/Path
            Catalogo galassie lens
        source_file : str/Path
            Catalogo galassie source (con shear)
        use_gpu : bool
            Usa GPU per calcoli (se disponibile)
        """
        self.lens_file = Path(lens_file)
        self.source_file = Path(source_file)
        self.use_gpu = use_gpu
        
        # Check GPU
        if use_gpu:
            try:
                import cupy as cp
                self.xp = cp
                print("✓ Usando GPU per calcoli")
            except ImportError:
                print("⚠️  cupy non disponibile, uso CPU")
                self.xp = np
                self.use_gpu = False
        else:
            self.xp = np
        
        # Carica dati
        print(f"\n📂 Caricamento dati...")
        
        # Carica da .npz (numpy) invece di .fits (astropy)
        lens_npz = np.load(self.lens_file)
        source_npz = np.load(self.source_file)
        
        # Converti in dict-like object per compatibilità
        self.lens = {key: lens_npz[key] for key in lens_npz.files}
        self.source = {key: source_npz[key] for key in source_npz.files}
        
        print(f"   Lens: {len(self.lens['ra'])} galassie")
        print(f"   Source: {len(self.source['ra'])} galassie")
    
    def create_mass_bins(self, n_bins=4):
        """
        Crea bin di massa stellare
        
        Returns
        -------
        bins : list of (log_mass_min, log_mass_max)
        """
        log_mass = self.lens['log_mass']
        
        # Bin uniformi in log-space
        mass_min, mass_max = log_mass.min(), log_mass.max()
        edges = np.linspace(mass_min, mass_max, n_bins + 1)
        
        bins = [(edges[i], edges[i+1]) for i in range(n_bins)]
        
        print(f"\n📊 Creazione {n_bins} bin di massa:")
        for i, (m_min, m_max) in enumerate(bins):
            n_gal = np.sum((log_mass >= m_min) & (log_mass < m_max))
            print(f"   Bin {i+1}: log(M*) = {m_min:.1f}-{m_max:.1f}, N={n_gal}")
        
        return bins
    
    def calc_angular_separation(self, ra1, dec1, ra2, dec2):
        """
        Calcola separazione angolare in gradi
        
        Usa formula haversine per precisione
        """
        ra1, dec1 = np.radians(ra1), np.radians(dec1)
        ra2, dec2 = np.radians(ra2), np.radians(dec2)
        
        dra = ra2 - ra1
        ddec = dec2 - dec1
        
        a = np.sin(ddec/2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return np.degrees(c)
    
    def calc_physical_separation(self, theta_deg, z_lens):
        """
        Converte separazione angolare in distanza fisica [kpc]
        
        Usa approssimazione small angle + cosmologia flat ΛCDM
        H0 = 70 km/s/Mpc
        """
        # Distanza angolare [Mpc] - approssimata per z < 1
        D_A = 3000 * z_lens / (1 + z_lens)  # Mpc (semplificato)
        
        # Separazione fisica
        r_phys = D_A * 1000 * np.radians(theta_deg)  # kpc
        
        return r_phys
    
    def stack_lensing_signal(self, lens_bin, radii_kpc):
        """
        Stacking del segnale di lensing per un bin di massa
        
        Parameters
        ----------
        lens_bin : Table
            Sottocampione di lens
        radii_kpc : array
            Bin radiali in kpc
        
        Returns
        -------
        DeltaSigma : array
            Densità superficiale in eccesso [M☉/kpc²]
        DeltaSigma_err : array
            Errore su ΔΣ
        """
        n_radii = len(radii_kpc) - 1
        DeltaSigma = np.zeros(n_radii)
        weights_sum = np.zeros(n_radii)
        
        n_lens_bin = len(lens_bin['ra'])
        print(f"   Stacking su {n_lens_bin} lens...")
        
        for i in tqdm(range(n_lens_bin), desc="Stacking"):
            lens = {k: v[i] for k, v in lens_bin.items()}
            # Trova source vicine
            theta = self.calc_angular_separation(
                lens['ra'], lens['dec'],
                self.source['ra'], self.source['dec']
            )
            
            # Converti in distanza fisica
            r_phys = self.calc_physical_separation(theta, lens['z'])
            
            # Seleziona source dietro la lens
            mask_z = self.source['z'] > lens['z'] + 0.1
            
            # Per ogni bin radiale
            for i in range(n_radii):
                r_min, r_max = radii_kpc[i], radii_kpc[i+1]
                mask_r = (r_phys >= r_min) & (r_phys < r_max) & mask_z
                
                if np.sum(mask_r) > 0:
                    # Shear tangenziale medio
                    gamma_t = np.mean(self.source['gamma_t'][mask_r])
                    
                    # Critical surface density (approssimato)
                    Sigma_crit = 1e15  # M☉/kpc² tipico, da raffinare
                    
                    # ΔΣ = Σ_crit × γ_t
                    DeltaSigma[i] += gamma_t * Sigma_crit
                    weights_sum[i] += 1
        
        # Media pesata
        mask_nonzero = weights_sum > 0
        DeltaSigma[mask_nonzero] /= weights_sum[mask_nonzero]
        
        # Errore (bootstrap o Jackknife, qui semplificato)
        # Aggiungi piccolo valore per evitare divisioni per zero
        DeltaSigma_err = np.abs(DeltaSigma) / np.sqrt(weights_sum + 1) + 1e10  # M☉/kpc²
        
        # Assicura che errori siano ragionevoli (almeno 10% del valore)
        DeltaSigma_err = np.maximum(DeltaSigma_err, np.abs(DeltaSigma) * 0.1 + 1e10)
        
        return DeltaSigma, DeltaSigma_err
    
    def predict_gcv(self, Mb_mean, radii_kpc):
        """
        Predizione GCV per ΔΣ(R)
        
        Con transizione da r⁻² a r⁻³
        
        Parameters
        ----------
        Mb_mean : float
            Massa barionica media del bin [M☉]
        radii_kpc : array
            Raggi in kpc
        
        Returns
        -------
        DeltaSigma_gcv : array
            Predizione ΔΣ(R) GCV [M☉/kpc²]
        """
        # Converte in SI
        Mb = Mb_mean * M_sun
        
        # Velocità piatta GCV
        v_inf = (G * Mb * A0)**(0.25)  # m/s
        
        # Raggio caratteristico
        Rc = np.sqrt(G * Mb / A0) / kpc  # kpc
        Rt = ALPHA * Rc  # kpc
        
        # Centri bin radiali
        r_centers = (radii_kpc[:-1] + radii_kpc[1:]) / 2
        
        # Profilo con transizione
        rho0_Rt = v_inf**2 / (4 * np.pi * G) / kpc  # M☉/kpc³
        
        # Densità superficiale proiettata (integrale linea di vista)
        # Per r << Rt: ΔΣ ~ v²/(4GR) ~ 1/R
        # Per r >> Rt: ΔΣ più ripido
        
        # Formula approssimata
        DeltaSigma_gcv = np.zeros_like(r_centers)
        
        for i, r in enumerate(r_centers):
            if r < Rt:
                # Regime SIS
                DeltaSigma_gcv[i] = v_inf**2 / (4 * G * r * kpc)
            else:
                # Regime transizione (pendenza più ripida)
                DeltaSigma_gcv[i] = v_inf**2 / (4 * G * Rt * kpc) * (Rt / r)**1.7
        
        # Converti in M☉/kpc²
        DeltaSigma_gcv /= M_sun / kpc**2
        
        return DeltaSigma_gcv
    
    def compare_profiles(self, r_centers, DeltaSigma_obs, DeltaSigma_err,
                        DeltaSigma_gcv, Mb_mean, save_plot=True):
        """
        Confronto osservato vs GCV
        
        Returns
        -------
        chi2 : float
            Chi-quadro ridotto
        p_value : float
            P-value del fit
        """
        # Chi-quadro (filtra valori non validi)
        mask_valid = (DeltaSigma_err > 0) & np.isfinite(DeltaSigma_obs) & np.isfinite(DeltaSigma_gcv)
        if np.sum(mask_valid) < 3:
            # Troppo pochi punti validi
            chi2_red, p_value = 999, 0.0
        else:
            chi2 = np.sum(((DeltaSigma_obs[mask_valid] - DeltaSigma_gcv[mask_valid]) / DeltaSigma_err[mask_valid])**2)
            dof = np.sum(mask_valid) - 1  # 1 parametro: Mb
            chi2_red = chi2 / max(dof, 1)
            p_value = 1 - stats.chi2.cdf(chi2, max(dof, 1))
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        # Profilo
        ax1.errorbar(r_centers, DeltaSigma_obs, yerr=DeltaSigma_err,
                     fmt='o', label='Osservato', color='black',
                     capsize=3, markersize=6)
        ax1.plot(r_centers, DeltaSigma_gcv, '-', label='GCV', 
                color='red', linewidth=2)
        
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_ylabel(r'$\Delta\Sigma$ [M$_\odot$/kpc$^2$]', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Titolo con info
        ax1.set_title(f'Mb = {Mb_mean:.2e} M☉ | χ²/dof = {chi2_red:.2f} | p = {p_value:.3f}',
                     fontsize=10)
        
        # Residui
        residuals = (DeltaSigma_obs - DeltaSigma_gcv) / DeltaSigma_err
        ax2.axhline(0, color='red', linestyle='--', linewidth=1)
        ax2.errorbar(r_centers, residuals, yerr=1.0,
                    fmt='o', color='black', capsize=3)
        ax2.set_xscale('log')
        ax2.set_xlabel('R [kpc]', fontsize=12)
        ax2.set_ylabel('Residui [σ]', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(2, color='gray', linestyle=':', alpha=0.5)
        ax2.axhline(-2, color='gray', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        
        if save_plot:
            plots_dir = Path(__file__).parent / 'plots'
            plots_dir.mkdir(exist_ok=True)
            filename = f'lensing_Mb{Mb_mean:.1e}.png'
            plt.savefig(plots_dir / filename, dpi=150, bbox_inches='tight')
            print(f"   💾 Plot salvato: plots/{filename}")
        
        plt.close()
        
        return chi2_red, p_value
    
    def run_full_analysis(self, n_bins=4, radii_kpc=None):
        """
        Analisi completa Test 2
        
        Returns
        -------
        results : dict
            Dizionario con tutti i risultati
        """
        if radii_kpc is None:
            # Bin radiali logaritmici 30-1000 kpc
            radii_kpc = np.logspace(np.log10(30), np.log10(1000), 15)
        
        r_centers = (radii_kpc[:-1] + radii_kpc[1:]) / 2
        
        print("\n" + "="*60)
        print("🔬 TEST 2: WEAK LENSING")
        print("="*60)
        
        # Crea bin massa
        mass_bins = self.create_mass_bins(n_bins)
        
        results = {
            'mass_bins': [],
            'Mb_mean': [],
            'chi2_red': [],
            'p_value': [],
            'verdict': []
        }
        
        # Per ogni bin di massa
        for i, (log_m_min, log_m_max) in enumerate(mass_bins):
            print(f"\n📊 Bin {i+1}/{n_bins}: log(M*) = {log_m_min:.1f}-{log_m_max:.1f}")
            
            # Seleziona lens nel bin
            mask = ((self.lens['log_mass'] >= log_m_min) & 
                   (self.lens['log_mass'] < log_m_max))
            
            # Filtra tutti i campi
            lens_bin = {key: val[mask] for key, val in self.lens.items()}
            
            # Massa media
            Mb_mean = 10**np.mean(lens_bin['log_mass'])
            print(f"   Mb medio: {Mb_mean:.2e} M☉")
            
            # Stacking
            print(f"   Stacking lensing signal...")
            DeltaSigma_obs, DeltaSigma_err = self.stack_lensing_signal(
                lens_bin, radii_kpc
            )
            
            # Predizione GCV
            print(f"   Calcolo predizione GCV...")
            DeltaSigma_gcv = self.predict_gcv(Mb_mean, radii_kpc)
            
            # Confronto
            print(f"   Confronto statistico...")
            chi2_red, p_value = self.compare_profiles(
                r_centers, DeltaSigma_obs, DeltaSigma_err,
                DeltaSigma_gcv, Mb_mean
            )
            
            # Verdetto
            if p_value > 0.05:
                verdict = "✅ COMPATIBILE"
            elif p_value > 0.01:
                verdict = "⚠️  TENSIONE"
            else:
                verdict = "❌ INCOMPATIBILE"
            
            print(f"\n   Risultati:")
            print(f"      χ²/dof = {chi2_red:.2f}")
            print(f"      p-value = {p_value:.3f}")
            print(f"      Verdetto: {verdict}")
            
            # Salva risultati
            results['mass_bins'].append((log_m_min, log_m_max))
            results['Mb_mean'].append(Mb_mean)
            results['chi2_red'].append(chi2_red)
            results['p_value'].append(p_value)
            results['verdict'].append(verdict)
        
        return results
    
    def final_verdict(self, results):
        """
        Verdetto finale Test 2
        """
        print("\n" + "="*60)
        print("🏁 VERDETTO FINALE TEST 2")
        print("="*60)
        
        n_bins = len(results['mass_bins'])
        n_pass = sum('✅' in v for v in results['verdict'])
        n_tension = sum('⚠️' in v for v in results['verdict'])
        n_fail = sum('❌' in v for v in results['verdict'])
        
        print(f"\nRisultati su {n_bins} bin di massa:")
        print(f"   ✅ Compatibili: {n_pass}/{n_bins}")
        print(f"   ⚠️  Tensione: {n_tension}/{n_bins}")
        print(f"   ❌ Incompatibili: {n_fail}/{n_bins}")
        
        print(f"\nDettagli:")
        for i, (mb, chi2, pval, verdict) in enumerate(zip(
            results['Mb_mean'], results['chi2_red'],
            results['p_value'], results['verdict']
        )):
            print(f"   Bin {i+1}: Mb={mb:.2e} M☉, χ²={chi2:.2f}, p={pval:.3f} → {verdict}")
        
        # Verdetto globale
        print(f"\n{'='*60}")
        
        if n_fail == 0 and n_tension <= 1:
            print("🎉 GCV SUPERA TEST 2 - WEAK LENSING")
            print("   Profilo con transizione compatibile con osservazioni")
            final = "PASS"
        elif n_fail <= 1 and n_pass >= n_bins//2:
            print("⚠️  GCV PARZIALMENTE COMPATIBILE")
            print("   Alcune tensioni ma non rigettata")
            final = "PARTIAL"
        else:
            print("❌ GCV FALLISCE TEST 2 - WEAK LENSING")
            print("   Troppi bin incompatibili")
            final = "FAIL"
        
        print(f"{'='*60}\n")
        
        return final

def main():
    """Main analisi"""
    import sys
    
    # Controlla file dati
    data_dir = Path(__file__).parent / 'data' / 'sdss'
    
    # Prova .npz prima, poi .fits
    lens_file = data_dir / 'mock_lens_catalog.npz'
    source_file = data_dir / 'mock_source_catalog.npz'
    
    if not lens_file.exists():
        lens_file = data_dir / 'mock_lens_catalog.fits'
        source_file = data_dir / 'mock_source_catalog.fits'
    
    if not lens_file.exists() or not source_file.exists():
        print("❌ File dati non trovati!")
        print("   Esegui prima: python generate_mock_simple.py")
        return 1
    
    # Analisi
    analyzer = LensingAnalysis(lens_file, source_file, use_gpu=True)
    results = analyzer.run_full_analysis(n_bins=4)
    final_verdict = analyzer.final_verdict(results)
    
    # Salva risultati
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    import json
    with open(results_dir / 'test2_lensing_results.json', 'w') as f:
        json.dump({
            'final_verdict': final_verdict,
            'chi2_red': results['chi2_red'],
            'p_value': results['p_value'],
            'verdicts': results['verdict']
        }, f, indent=2)
    
    print(f"💾 Risultati salvati in: results/test2_lensing_results.json")
    
    return 0 if final_verdict == "PASS" else 1

if __name__ == '__main__':
    sys.exit(main())
