#!/usr/bin/env python3
"""
GCV con Profilo Realistico

Il test precedente mostra che il profilo isotermo semplificato
non funziona (normalizzazione ~0.45 invece di ~1).

Questo test usa un profilo barionico REALISTICO (Hernquist)
e calcola Delta Sigma correttamente.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.interpolate import interp1d

print("="*70)
print("GCV con Profilo Barionico Realistico")
print("="*70)

# Directories
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

# Costanti
G = 6.674e-11
M_sun = 1.989e30
kpc = 3.086e19
pc = 3.086e16
H0 = 67.4
h = H0 / 100

# Parametri GCV
GCV_PARAMS = {
    'a0': 1.80e-10,
    'A0': 1.16,
    'gamma': 0.06,
    'beta': 0.90,
}

ALPHA_DERIVED = 0.5

# Dati SDSS
SDSS_DATA = {
    'L4': {
        'M_stellar': 1.5e11,
        'R_kpc': np.array([28, 44, 70, 111, 176, 279, 442, 700, 1109, 1757]) / h,
        'DeltaSigma': np.array([127.3, 98.2, 74.1, 55.8, 40.2, 27.1, 17.8, 11.2, 7.1, 4.3]),
        'error': np.array([11.2, 8.5, 6.4, 4.8, 3.5, 2.4, 1.6, 1.1, 0.8, 0.6]),
    },
    'L2': {
        'M_stellar': 5e10,
        'R_kpc': np.array([28, 44, 70, 111, 176, 279, 442, 700, 1109]) / h,
        'DeltaSigma': np.array([68.5, 54.2, 41.3, 31.8, 23.1, 15.7, 10.2, 6.4, 4.0]),
        'error': np.array([7.8, 5.9, 4.4, 3.3, 2.4, 1.7, 1.2, 0.9, 0.7]),
    }
}

# =============================================================================
# PROFILO BARIONICO REALISTICO
# =============================================================================

def hernquist_rho(r_kpc, M_star, a_kpc):
    """Densita' 3D di Hernquist"""
    M = M_star * M_sun
    a = a_kpc * kpc
    r = max(r_kpc, 0.01) * kpc
    return M / (2 * np.pi) * a / (r * (r + a)**3)

def hernquist_sigma(R_kpc, M_star, a_kpc, r_max=5000):
    """Densita' superficiale di Hernquist (proiezione analitica)"""
    # Formula analitica per Sigma di Hernquist
    M = M_star * M_sun
    a = a_kpc * kpc
    R = R_kpc * kpc
    
    s = R / a
    if s < 1:
        X = np.sqrt(1 - s**2)
        Sigma = M / (2 * np.pi * a**2) * (1 / (s**2 - 1)**2) * \
                (-3 + (2 + s**2) * np.arccosh(1/s) / X)
    elif s > 1:
        X = np.sqrt(s**2 - 1)
        Sigma = M / (2 * np.pi * a**2) * (1 / (s**2 - 1)**2) * \
                (-3 + (2 + s**2) * np.arccos(1/s) / X)
    else:
        Sigma = M / (2 * np.pi * a**2) * (4/15)
    
    return Sigma  # kg/m^2

def chi_v(r_kpc, M_star, params):
    """Suscettibilita' GCV"""
    M = M_star * M_sun
    Lc = np.sqrt(G * M / params['a0']) / kpc
    chi = params['A0'] * (M_star / 1e11)**params['gamma'] * (1 + (r_kpc / Lc)**params['beta'])
    return chi

def compute_sigma_gcv(R_kpc, M_star, a_kpc, params, r_max=3000):
    """
    Sigma GCV = integral rho_b * (1 + chi_v) dz
    
    Con formula derivata: usiamo (1 + chi_v)^0.5 per Delta Sigma
    """
    def integrand(r):
        if r <= R_kpc:
            return 0
        rho = hernquist_rho(r, M_star, a_kpc)
        chi = chi_v(r, M_star, params)
        # Usiamo (1 + chi)^0.5 come derivato
        return rho * (1 + chi)**ALPHA_DERIVED * r / np.sqrt(r**2 - R_kpc**2)
    
    result, _ = quad(integrand, R_kpc * 1.001, r_max, limit=200)
    return 2 * result * kpc

def compute_delta_sigma(R_array, M_star, a_kpc, params, use_gcv=True):
    """Calcola Delta Sigma = Sigma_mean(<R) - Sigma(R)"""
    
    # Calcola Sigma su griglia fine
    R_fine = np.logspace(np.log10(max(1, R_array.min()/2)), 
                         np.log10(R_array.max()*1.5), 100)
    
    if use_gcv:
        Sigma_fine = np.array([compute_sigma_gcv(R, M_star, a_kpc, params) for R in R_fine])
    else:
        Sigma_fine = np.array([hernquist_sigma(R, M_star, a_kpc) for R in R_fine])
    
    # Interpola
    Sigma_interp = interp1d(R_fine, Sigma_fine, kind='cubic', fill_value='extrapolate')
    
    # Calcola Delta Sigma per ogni R
    Delta_Sigma = np.zeros_like(R_array, dtype=float)
    
    for i, R in enumerate(R_array):
        # Sigma(R)
        Sigma_R = Sigma_interp(R)
        
        # Sigma_mean(<R) = (2/R^2) * integral_0^R Sigma(R') R' dR'
        R_inner = np.linspace(R_fine[0], R, 50)
        Sigma_inner = Sigma_interp(R_inner)
        integral = np.trapz(Sigma_inner * R_inner, R_inner)
        Sigma_mean = 2 * integral / R**2
        
        Delta_Sigma[i] = Sigma_mean - Sigma_R
    
    # Converti in M_sun/pc^2
    return Delta_Sigma / (M_sun / pc**2)

# =============================================================================
# TEST
# =============================================================================
print("\n" + "="*70)
print("CALCOLO DELTA SIGMA CON PROFILO REALISTICO")
print("="*70)

results = {}

for sample_name, data in SDSS_DATA.items():
    print(f"\n--- Sample {sample_name} ---")
    
    M_star = data['M_stellar']
    R = data['R_kpc']
    obs = data['DeltaSigma']
    err = data['error']
    
    # Scala di Hernquist (relazione empirica)
    a_kpc = 3.0 * (M_star / 1e11)**0.3
    
    print(f"  M* = {M_star:.1e} M_sun")
    print(f"  a = {a_kpc:.1f} kpc")
    print(f"  Lc = {np.sqrt(G * M_star * M_sun / GCV_PARAMS['a0']) / kpc:.1f} kpc")
    
    # Calcola Delta Sigma GCV
    print("  Calcolando Delta Sigma GCV...")
    DS_gcv = compute_delta_sigma(R, M_star, a_kpc, GCV_PARAMS, use_gcv=True)
    
    # Calcola Delta Sigma barionico
    print("  Calcolando Delta Sigma barionico...")
    DS_b = compute_delta_sigma(R, M_star, a_kpc, GCV_PARAMS, use_gcv=False)
    
    # Fit normalizzazione
    def chi2_func(norm):
        pred = norm * DS_gcv
        return np.sum(((obs - pred) / err)**2)
    
    res = minimize(chi2_func, [1.0], method='Nelder-Mead')
    norm_best = res.x[0]
    chi2_best = res.fun
    
    print(f"\n  Risultati:")
    print(f"    Normalizzazione ottimale: {norm_best:.3f}")
    print(f"    Chi2 = {chi2_best:.2f}")
    print(f"    Chi2/dof = {chi2_best/(len(R)-1):.3f}")
    
    # Confronto punto per punto
    print(f"\n    {'R':>8} {'Obs':>8} {'GCV':>8} {'Ratio':>8}")
    print("    " + "-"*40)
    for i in range(min(5, len(R))):
        pred = norm_best * DS_gcv[i]
        ratio = pred / obs[i] if obs[i] > 0 else 0
        print(f"    {R[i]:>8.0f} {obs[i]:>8.1f} {pred:>8.1f} {ratio:>8.2f}")
    
    results[sample_name] = {
        'norm': float(norm_best),
        'chi2': float(chi2_best),
        'chi2_red': float(chi2_best / (len(R) - 1)),
        'DS_gcv': DS_gcv.tolist(),
        'DS_b': DS_b.tolist()
    }

# =============================================================================
# CONFRONTO CON LCDM
# =============================================================================
print("\n" + "="*70)
print("CONFRONTO CON LCDM")
print("="*70)

chi2_gcv_tot = sum(r['chi2'] for r in results.values())
chi2_lcdm_tot = 27.79  # dal test precedente

N_tot = sum(len(SDSS_DATA[s]['R_kpc']) for s in SDSS_DATA)
k_gcv = 2  # normalizzazioni
k_lcdm = 4

AIC_gcv = chi2_gcv_tot + 2 * k_gcv
AIC_lcdm = chi2_lcdm_tot + 2 * k_lcdm
Delta_AIC = AIC_gcv - AIC_lcdm

print(f"\nGCV (profilo realistico + formula derivata):")
print(f"  Chi2 totale = {chi2_gcv_tot:.2f}")
print(f"  Chi2/dof = {chi2_gcv_tot/(N_tot-k_gcv):.3f}")

print(f"\nLCDM:")
print(f"  Chi2 totale = {chi2_lcdm_tot:.2f}")
print(f"  Chi2/dof = {chi2_lcdm_tot/(N_tot-k_lcdm):.3f}")

print(f"\nDelta AIC = {Delta_AIC:.1f}")

if Delta_AIC < -2:
    verdict = "GCV FAVORITA"
elif abs(Delta_AIC) < 2:
    verdict = "EQUIVALENTI"
elif Delta_AIC < 10:
    verdict = "LCDM FAVORITA"
else:
    verdict = "LCDM FORTEMENTE FAVORITA"

print(f"VERDETTO: {verdict}")

# =============================================================================
# ANALISI: Perche' la normalizzazione non e' 1?
# =============================================================================
print("\n" + "="*70)
print("ANALISI DELLA NORMALIZZAZIONE")
print("="*70)

print(f"""
Le normalizzazioni ottimali sono:
  L4: {results['L4']['norm']:.3f}
  L2: {results['L2']['norm']:.3f}

Se fossero ~1, significherebbe che la formula derivata e' corretta.
Se sono diverse da 1, ci sono due possibilita':

1. Il profilo base (Hernquist) non e' adatto per queste galassie
2. La formula derivata ha bisogno di correzioni

Verifichiamo confrontando con il rapporto atteso:
""")

for sample_name, data in SDSS_DATA.items():
    M_star = data['M_stellar']
    R_mid = data['R_kpc'][len(data['R_kpc'])//2]
    chi = chi_v(R_mid, M_star, GCV_PARAMS)
    
    expected_boost = (1 + chi)**ALPHA_DERIVED
    actual_boost = data['DeltaSigma'][len(data['R_kpc'])//2] / results[sample_name]['DS_b'][len(data['R_kpc'])//2]
    
    print(f"\n{sample_name} a R={R_mid:.0f} kpc:")
    print(f"  chi_v = {chi:.1f}")
    print(f"  Boost atteso (1+chi)^0.5 = {expected_boost:.1f}")
    print(f"  Boost osservato DS_obs/DS_b = {actual_boost:.1f}")
    print(f"  Ratio = {actual_boost/expected_boost:.2f}")

# =============================================================================
# PLOT
# =============================================================================
print("\n" + "="*70)
print("GENERAZIONE PLOT")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('GCV Profilo Realistico + Formula Derivata', fontsize=13, fontweight='bold')

for idx, (sample_name, data) in enumerate(SDSS_DATA.items()):
    ax = axes[idx]
    
    R = data['R_kpc']
    obs = data['DeltaSigma']
    err = data['error']
    
    norm = results[sample_name]['norm']
    DS_gcv = np.array(results[sample_name]['DS_gcv'])
    DS_b = np.array(results[sample_name]['DS_b'])
    
    ax.errorbar(R, obs, yerr=err, fmt='ko', capsize=3, label='SDSS DR7', markersize=7)
    ax.plot(R, norm * DS_gcv, 'b-', linewidth=2, label=f'GCV (norm={norm:.2f})')
    ax.plot(R, DS_b, 'g--', linewidth=1, alpha=0.5, label='Solo barioni')
    
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('Delta Sigma [M_sun/pc^2]')
    ax.set_title(f'{sample_name}: chi2/dof={results[sample_name]["chi2_red"]:.2f}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'gcv_realistic_profile.png', dpi=300, bbox_inches='tight')
print("Plot salvato")

# Salva
final_results = {
    'test': 'GCV Realistic Profile',
    'alpha_derived': ALPHA_DERIVED,
    'samples': results,
    'chi2_gcv_tot': float(chi2_gcv_tot),
    'chi2_lcdm_tot': float(chi2_lcdm_tot),
    'Delta_AIC': float(Delta_AIC),
    'verdict': verdict
}

with open(RESULTS_DIR / 'gcv_realistic_profile_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print("\n" + "="*70)
print("CONCLUSIONE")
print("="*70)
print(f"""
Il profilo realistico (Hernquist) con formula derivata (alpha=0.5)
da' chi2/dof ~ {chi2_gcv_tot/(N_tot-k_gcv):.1f}

Questo e' ancora peggio di LCDM (chi2/dof ~ 1.9).

PROBLEMA IDENTIFICATO:
La normalizzazione ~{np.mean([r['norm'] for r in results.values()]):.2f} indica che
GCV SOTTOSTIMA il segnale di lensing di un fattore ~2.

POSSIBILI SOLUZIONI:
1. Aumentare A0 (ampiezza di chi_v)
2. Modificare la forma di chi_v per il lensing
3. Considerare che i dati SDSS includono anche alone di materia oscura
""")
print("="*70)
