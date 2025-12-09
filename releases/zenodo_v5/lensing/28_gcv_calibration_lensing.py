#!/usr/bin/env python3
"""
CALIBRAZIONE GCV PER LENSING

I test precedenti mostrano che:
1. alpha = 0.5 e' derivato correttamente dalla matematica di Delta Sigma
2. Ma il boost (1+chi_v)^0.5 non e' sufficiente per spiegare i dati

Questo test trova quale valore di A0 (ampiezza di chi_v) serve
per fittare i dati di lensing, e lo confronta con quello da rotation curves.

Se A0_lensing ~ A0_rotation, GCV e' consistente.
Se sono molto diversi, c'e' un problema.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
from scipy.interpolate import interp1d

print("="*70)
print("CALIBRAZIONE GCV PER LENSING")
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

# Parametri GCV BASE (da rotation curves)
GCV_BASE = {
    'a0': 1.80e-10,
    'A0': 1.16,  # QUESTO vogliamo calibrare
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
# MODELLO GCV SEMPLIFICATO MA CORRETTO
# =============================================================================

def chi_v(R_kpc, M_star, A0, params):
    """Suscettibilita' GCV con A0 variabile"""
    M = M_star * M_sun
    Lc = np.sqrt(G * M / params['a0']) / kpc
    chi = A0 * (M_star / 1e11)**params['gamma'] * (1 + (R_kpc / Lc)**params['beta'])
    return chi

def delta_sigma_gcv(R_kpc, M_star, A0, params):
    """
    Delta Sigma GCV semplificato
    
    Usiamo la formula empirica che funziona:
    DS = DS_base * (1 + chi_v)^alpha
    
    dove DS_base viene dalla velocita' asintotica
    """
    M = M_star * M_sun
    v_inf = (G * M * params['a0'])**(0.25)
    
    R_m = R_kpc * kpc
    ds_base = v_inf**2 / (4 * G * R_m)
    ds_base_Msun_pc2 = ds_base / (M_sun / pc**2)
    
    chi = chi_v(R_kpc, M_star, A0, params)
    
    # Formula derivata
    ds = ds_base_Msun_pc2 * (1 + chi)**ALPHA_DERIVED
    
    return ds

# =============================================================================
# CALIBRAZIONE A0
# =============================================================================
print("\n" + "="*70)
print("CALIBRAZIONE A0 SU DATI LENSING")
print("="*70)

def chi2_total(A0):
    """Chi2 totale per dato A0"""
    chi2 = 0
    for name, data in SDSS_DATA.items():
        R = data['R_kpc']
        obs = data['DeltaSigma']
        err = data['error']
        M_star = data['M_stellar']
        
        pred = delta_sigma_gcv(R, M_star, A0, GCV_BASE)
        chi2 += np.sum(((obs - pred) / err)**2)
    
    return chi2

# Cerca A0 ottimale
print("\nCercando A0 ottimale...")
result = minimize(lambda x: chi2_total(x[0]), [10], method='Nelder-Mead')
A0_best = result.x[0]
chi2_best = result.fun

N_tot = sum(len(d['R_kpc']) for d in SDSS_DATA.values())

print(f"\nRisultati:")
print(f"  A0 da rotation curves: {GCV_BASE['A0']}")
print(f"  A0 ottimale per lensing: {A0_best:.2f}")
print(f"  Ratio: {A0_best / GCV_BASE['A0']:.1f}x")
print(f"  Chi2 = {chi2_best:.2f}")
print(f"  Chi2/dof = {chi2_best/(N_tot-1):.3f}")

# =============================================================================
# CONFRONTO CON LCDM
# =============================================================================
print("\n" + "="*70)
print("CONFRONTO CON LCDM")
print("="*70)

chi2_lcdm = 27.79
k_lcdm = 4
k_gcv = 1  # solo A0

AIC_gcv = chi2_best + 2 * k_gcv
AIC_lcdm = chi2_lcdm + 2 * k_lcdm
Delta_AIC = AIC_gcv - AIC_lcdm

print(f"\nGCV (A0 calibrato):")
print(f"  Chi2 = {chi2_best:.2f}")
print(f"  Parametri = {k_gcv}")
print(f"  AIC = {AIC_gcv:.1f}")

print(f"\nLCDM:")
print(f"  Chi2 = {chi2_lcdm:.2f}")
print(f"  Parametri = {k_lcdm}")
print(f"  AIC = {AIC_lcdm:.1f}")

print(f"\nDelta AIC = {Delta_AIC:.1f}")

if Delta_AIC < -10:
    verdict = "GCV FORTEMENTE FAVORITA"
elif Delta_AIC < -2:
    verdict = "GCV FAVORITA"
elif abs(Delta_AIC) < 2:
    verdict = "EQUIVALENTI"
elif Delta_AIC < 10:
    verdict = "LCDM FAVORITA"
else:
    verdict = "LCDM FORTEMENTE FAVORITA"

print(f"VERDETTO: {verdict}")

# =============================================================================
# ANALISI DETTAGLIATA
# =============================================================================
print("\n" + "="*70)
print("ANALISI DETTAGLIATA")
print("="*70)

print("\nConfronto predizioni con A0 calibrato:")

for name, data in SDSS_DATA.items():
    print(f"\n{name}:")
    R = data['R_kpc']
    obs = data['DeltaSigma']
    err = data['error']
    M_star = data['M_stellar']
    
    pred = delta_sigma_gcv(R, M_star, A0_best, GCV_BASE)
    
    print(f"  {'R':>8} {'Obs':>8} {'Pred':>8} {'Res/Err':>8}")
    print("  " + "-"*40)
    for i in range(len(R)):
        res = (obs[i] - pred[i]) / err[i]
        print(f"  {R[i]:>8.0f} {obs[i]:>8.1f} {pred[i]:>8.1f} {res:>8.2f}")

# =============================================================================
# INTERPRETAZIONE
# =============================================================================
print("\n" + "="*70)
print("INTERPRETAZIONE")
print("="*70)

print(f"""
RISULTATO CHIAVE:

Per fittare i dati di lensing, serve A0 = {A0_best:.1f}
invece di A0 = {GCV_BASE['A0']} da rotation curves.

Questo e' un fattore {A0_best/GCV_BASE['A0']:.0f}x piu' grande!

POSSIBILI INTERPRETAZIONI:

1. GCV HA BISOGNO DI CALIBRAZIONE SEPARATA per lensing
   - Simile a come MOND ha bisogno di calibrazione per cluster
   - Non necessariamente un problema fatale

2. LA FORMULA DERIVATA (alpha=0.5) E' CORRETTA
   - Ma i parametri base (A0) devono essere diversi
   - Questo potrebbe indicare fisica diversa

3. I DATI SDSS INCLUDONO EFFETTI NON MODELLATI
   - Contributo da gas caldo
   - Effetti di ambiente (clustering)
   - Sistematici nei dati

4. POSSIBILE SOLUZIONE TEORICA:
   - Se chi_v per lensing = chi_v_dyn^2 invece di chi_v_dyn
   - Allora (1 + chi_v_dyn^2)^0.5 ~ chi_v_dyn per chi_v >> 1
   - Questo darebbe il boost necessario!
""")

# =============================================================================
# TEST: chi_v^2 invece di chi_v
# =============================================================================
print("\n" + "="*70)
print("TEST ALTERNATIVO: chi_v^2")
print("="*70)

def delta_sigma_gcv_squared(R_kpc, M_star, params):
    """Delta Sigma con chi_v^2"""
    M = M_star * M_sun
    v_inf = (G * M * params['a0'])**(0.25)
    
    R_m = R_kpc * kpc
    ds_base = v_inf**2 / (4 * G * R_m)
    ds_base_Msun_pc2 = ds_base / (M_sun / pc**2)
    
    chi = chi_v(R_kpc, M_star, params['A0'], params)
    
    # Prova con chi^2
    ds = ds_base_Msun_pc2 * (1 + chi**2)**ALPHA_DERIVED
    
    return ds

def chi2_squared(norm):
    chi2 = 0
    for name, data in SDSS_DATA.items():
        R = data['R_kpc']
        obs = data['DeltaSigma']
        err = data['error']
        M_star = data['M_stellar']
        
        pred = norm * delta_sigma_gcv_squared(R, M_star, GCV_BASE)
        chi2 += np.sum(((obs - pred) / err)**2)
    
    return chi2

res_sq = minimize(lambda x: chi2_squared(x[0]), [1.0], method='Nelder-Mead')
norm_sq = res_sq.x[0]
chi2_sq = res_sq.fun

print(f"\nCon formula (1 + chi_v^2)^0.5:")
print(f"  Normalizzazione = {norm_sq:.3f}")
print(f"  Chi2 = {chi2_sq:.2f}")
print(f"  Chi2/dof = {chi2_sq/(N_tot-1):.3f}")

if chi2_sq < chi2_best:
    print(f"\n  MIGLIORE della formula lineare!")
else:
    print(f"\n  Peggiore della formula lineare")

# =============================================================================
# PLOT FINALE
# =============================================================================
print("\n" + "="*70)
print("GENERAZIONE PLOT")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'GCV Calibrato (A0={A0_best:.1f}) vs LCDM', fontsize=13, fontweight='bold')

for idx, (name, data) in enumerate(SDSS_DATA.items()):
    ax = axes[idx]
    
    R = data['R_kpc']
    obs = data['DeltaSigma']
    err = data['error']
    M_star = data['M_stellar']
    
    # Dati
    ax.errorbar(R, obs, yerr=err, fmt='ko', capsize=3, label='SDSS DR7', markersize=7)
    
    # GCV calibrato
    R_plot = np.logspace(np.log10(R.min()), np.log10(R.max()), 100)
    pred_gcv = delta_sigma_gcv(R_plot, M_star, A0_best, GCV_BASE)
    ax.plot(R_plot, pred_gcv, 'b-', linewidth=2, label=f'GCV (A0={A0_best:.1f})')
    
    # GCV originale
    pred_orig = delta_sigma_gcv(R_plot, M_star, GCV_BASE['A0'], GCV_BASE)
    ax.plot(R_plot, pred_orig, 'b--', linewidth=1, alpha=0.5, label=f'GCV orig (A0={GCV_BASE["A0"]})')
    
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('Delta Sigma [M_sun/pc^2]')
    ax.set_title(f'{name}: M*={M_star:.1e}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'gcv_calibration_lensing.png', dpi=300, bbox_inches='tight')
print("Plot salvato")

# Salva risultati
results = {
    'test': 'GCV Calibration for Lensing',
    'A0_rotation': GCV_BASE['A0'],
    'A0_lensing': float(A0_best),
    'ratio': float(A0_best / GCV_BASE['A0']),
    'chi2_gcv': float(chi2_best),
    'chi2_lcdm': float(chi2_lcdm),
    'Delta_AIC': float(Delta_AIC),
    'verdict': verdict,
    'chi2_squared_test': float(chi2_sq)
}

with open(RESULTS_DIR / 'gcv_calibration_lensing.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("CONCLUSIONE FINALE")
print("="*70)
print(f"""
GCV puo' fittare i dati di lensing SE:
- A0 viene aumentato da {GCV_BASE['A0']} a {A0_best:.1f} (fattore {A0_best/GCV_BASE['A0']:.0f}x)
- Oppure si usa una forma diversa di chi_v

Con A0 calibrato:
- Chi2/dof = {chi2_best/(N_tot-1):.2f}
- Delta AIC vs LCDM = {Delta_AIC:.1f}
- Verdetto: {verdict}

IMPLICAZIONE:
GCV richiede parametri DIVERSI per rotation curves e lensing.
Questo e' simile al problema di MOND con i cluster.

POSSIBILE SOLUZIONE:
Trovare una forma unificata di chi_v che funzioni per entrambi.
""")
print("="*70)
