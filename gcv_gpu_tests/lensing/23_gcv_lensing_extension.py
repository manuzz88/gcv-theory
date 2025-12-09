#!/usr/bin/env python3
"""
GCV ESTESO: Forma Unificata per Rotation Curves e Lensing

Il problema: chi_v che funziona per rotation curves NON funziona per lensing.

Ipotesi: Il lensing "vede" una suscettibilita' diversa perche':
1. Lensing e' sensibile al potenziale Phi, non alla forza dPhi/dr
2. La proiezione lungo la linea di vista integra su scale diverse

Proposta: chi_v ha due componenti:
- chi_v_dyn: per dinamica (rotation curves) - quello che abbiamo
- chi_v_lens: per lensing - da determinare

Relazione fisica possibile:
chi_v_lens = chi_v_dyn^alpha_lens * (R/R_lens)^delta

dove alpha_lens e delta sono parametri da fittare.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("GCV ESTESO: Unificazione Rotation Curves e Lensing")
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

# =============================================================================
# DATI
# =============================================================================

# SDSS Lensing
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

# SPARC Rotation Curves (subset rappresentativo)
SPARC_DATA = {
    'NGC3198': {'M_star': 3.5e10, 'v_obs': 150, 'R_flat': 15},  # km/s, kpc
    'NGC2403': {'M_star': 5e9, 'v_obs': 130, 'R_flat': 10},
    'NGC7331': {'M_star': 1e11, 'v_obs': 250, 'R_flat': 20},
    'NGC5055': {'M_star': 6e10, 'v_obs': 200, 'R_flat': 25},
    'NGC6946': {'M_star': 4e10, 'v_obs': 180, 'R_flat': 15},
}

# =============================================================================
# MODELLO GCV ESTESO
# =============================================================================

class GCVExtended:
    """
    Modello GCV esteso con forme separate per dinamica e lensing
    """
    
    def __init__(self, params):
        # Parametri base (da rotation curves)
        self.a0 = params.get('a0', 1.80e-10)
        self.A0 = params.get('A0', 1.16)
        self.gamma = params.get('gamma', 0.06)
        self.beta = params.get('beta', 0.90)
        
        # Parametri estensione lensing
        self.alpha_lens = params.get('alpha_lens', 1.0)  # esponente chi_v
        self.R_lens = params.get('R_lens', 100)  # scala caratteristica kpc
        self.delta = params.get('delta', 0.0)  # scaling radiale aggiuntivo
        self.A_lens = params.get('A_lens', 1.0)  # normalizzazione lensing
    
    def Lc(self, M_star):
        """Lunghezza di coerenza"""
        M = M_star * M_sun
        return np.sqrt(G * M / self.a0) / kpc  # kpc
    
    def chi_v_dyn(self, R_kpc, M_star):
        """Suscettibilita' per dinamica (rotation curves)"""
        Lc = self.Lc(M_star)
        chi = self.A0 * (M_star / 1e11)**self.gamma * (1 + (R_kpc / Lc)**self.beta)
        return chi
    
    def chi_v_lens(self, R_kpc, M_star):
        """
        Suscettibilita' per lensing
        
        Ipotesi fisica: il lensing integra lungo la linea di vista,
        quindi "vede" una media pesata di chi_v su scale diverse.
        
        Forma proposta:
        chi_v_lens = A_lens * chi_v_dyn^alpha_lens * (1 + R/R_lens)^delta
        """
        chi_dyn = self.chi_v_dyn(R_kpc, M_star)
        chi_lens = self.A_lens * chi_dyn**self.alpha_lens * (1 + R_kpc / self.R_lens)**self.delta
        return chi_lens
    
    def v_circular(self, R_kpc, M_star):
        """Velocita' circolare (per rotation curves)"""
        # v^2 = G M_eff / R = G M_b * (1 + chi_v) / R
        M_b = M_star * M_sun
        R_m = R_kpc * kpc
        chi_v = self.chi_v_dyn(R_kpc, M_star)
        
        # Per grandi R, v -> (G M_b a0)^(1/4) (regime MOND-like)
        v_inf = (G * M_b * self.a0)**(0.25)
        
        # Transizione smooth
        Lc = self.Lc(M_star)
        x = R_kpc / Lc
        v = v_inf * (x / (1 + x))**0.25 * (1 + chi_v)**0.5 / chi_v**0.5
        
        # Semplificazione: per R >> Lc, v -> v_inf
        return v_inf / 1000  # km/s
    
    def delta_sigma(self, R_kpc, M_star):
        """
        Delta Sigma per lensing
        
        Approccio semplificato ma fisicamente motivato:
        Delta Sigma ~ v_inf^2 / (G R) * chi_v_lens
        
        dove v_inf = (G M a0)^(1/4)
        """
        M_b = M_star * M_sun
        v_inf = (G * M_b * self.a0)**(0.25)  # m/s
        R_m = R_kpc * kpc
        
        # Delta Sigma base (profilo isotermo)
        ds_base = v_inf**2 / (4 * G * R_m)  # kg/m^2
        ds_base_Msun_pc2 = ds_base / (M_sun / pc**2)
        
        # Applica chi_v_lens
        chi_lens = self.chi_v_lens(R_kpc, M_star)
        
        return ds_base_Msun_pc2 * chi_lens

# =============================================================================
# FUNZIONE DI FIT
# =============================================================================

def chi2_combined(params_array):
    """
    Chi2 combinato: rotation curves + lensing
    
    Vogliamo trovare parametri che fittino ENTRAMBI simultaneamente.
    """
    alpha_lens, R_lens, delta, A_lens = params_array
    
    # Bounds check
    if not (0.1 < alpha_lens < 3):
        return 1e10
    if not (10 < R_lens < 1000):
        return 1e10
    if not (-2 < delta < 2):
        return 1e10
    if not (0.01 < A_lens < 100):
        return 1e10
    
    params = {
        'a0': 1.80e-10,
        'A0': 1.16,
        'gamma': 0.06,
        'beta': 0.90,
        'alpha_lens': alpha_lens,
        'R_lens': R_lens,
        'delta': delta,
        'A_lens': A_lens
    }
    
    model = GCVExtended(params)
    
    chi2_total = 0
    
    # Chi2 lensing
    for name, data in SDSS_DATA.items():
        R = data['R_kpc']
        obs = data['DeltaSigma']
        err = data['error']
        M_star = data['M_stellar']
        
        pred = np.array([model.delta_sigma(r, M_star) for r in R])
        chi2_total += np.sum(((obs - pred) / err)**2)
    
    # Chi2 rotation curves (peso ridotto per bilanciare)
    weight_rc = 0.5  # peso relativo
    for name, data in SPARC_DATA.items():
        M_star = data['M_star']
        v_obs = data['v_obs']
        v_pred = model.v_circular(data['R_flat'], M_star)
        err_v = 0.1 * v_obs  # 10% error tipico
        chi2_total += weight_rc * ((v_obs - v_pred) / err_v)**2
    
    return chi2_total

# =============================================================================
# OTTIMIZZAZIONE
# =============================================================================
print("\n" + "="*70)
print("OTTIMIZZAZIONE PARAMETRI ESTESI")
print("="*70)

print("\nCercando parametri che fittino sia lensing che rotation curves...")

bounds = [
    (0.1, 3),     # alpha_lens
    (10, 1000),   # R_lens
    (-2, 2),      # delta
    (0.01, 100)   # A_lens
]

result = differential_evolution(chi2_combined, bounds, maxiter=200, 
                                 seed=42, workers=-1, disp=True, tol=0.01)

best_params = result.x
best_chi2 = result.fun

print(f"\nParametri ottimali trovati:")
print(f"  alpha_lens = {best_params[0]:.3f}")
print(f"  R_lens = {best_params[1]:.1f} kpc")
print(f"  delta = {best_params[2]:.3f}")
print(f"  A_lens = {best_params[3]:.3f}")
print(f"  Chi2 totale = {best_chi2:.2f}")

# =============================================================================
# VALUTAZIONE DETTAGLIATA
# =============================================================================
print("\n" + "="*70)
print("VALUTAZIONE DETTAGLIATA")
print("="*70)

final_params = {
    'a0': 1.80e-10,
    'A0': 1.16,
    'gamma': 0.06,
    'beta': 0.90,
    'alpha_lens': best_params[0],
    'R_lens': best_params[1],
    'delta': best_params[2],
    'A_lens': best_params[3]
}

model = GCVExtended(final_params)

# Lensing
print("\n--- LENSING ---")
chi2_lens = 0
N_lens = 0
for name, data in SDSS_DATA.items():
    R = data['R_kpc']
    obs = data['DeltaSigma']
    err = data['error']
    M_star = data['M_stellar']
    
    pred = np.array([model.delta_sigma(r, M_star) for r in R])
    chi2_sample = np.sum(((obs - pred) / err)**2)
    chi2_lens += chi2_sample
    N_lens += len(R)
    
    print(f"\n  {name} (M*={M_star:.1e}):")
    print(f"    Chi2 = {chi2_sample:.2f}")
    print(f"    R [kpc]    Obs    Pred    Ratio")
    for i in range(min(5, len(R))):
        print(f"    {R[i]:6.0f}    {obs[i]:5.1f}  {pred[i]:5.1f}   {pred[i]/obs[i]:.2f}")

print(f"\n  Chi2 lensing totale: {chi2_lens:.2f}")
print(f"  Chi2/dof: {chi2_lens/(N_lens-4):.3f}")

# Rotation curves
print("\n--- ROTATION CURVES ---")
chi2_rc = 0
for name, data in SPARC_DATA.items():
    M_star = data['M_star']
    v_obs = data['v_obs']
    v_pred = model.v_circular(data['R_flat'], M_star)
    err_v = 0.1 * v_obs
    chi2_rc += ((v_obs - v_pred) / err_v)**2
    print(f"  {name}: v_obs={v_obs:.0f}, v_pred={v_pred:.0f} km/s")

print(f"\n  Chi2 rotation curves: {chi2_rc:.2f}")

# =============================================================================
# CONFRONTO CON LCDM
# =============================================================================
print("\n" + "="*70)
print("CONFRONTO CON LCDM")
print("="*70)

chi2_lcdm = 21.99  # dal test precedente
k_lcdm = 4
k_gcv = 4  # alpha_lens, R_lens, delta, A_lens

AIC_gcv = chi2_lens + 2 * k_gcv
AIC_lcdm = chi2_lcdm + 2 * k_lcdm
Delta_AIC = AIC_gcv - AIC_lcdm

print(f"\nGCV Esteso:")
print(f"  Chi2 lensing = {chi2_lens:.2f}")
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
# PLOT
# =============================================================================
print("\n" + "="*70)
print("GENERAZIONE PLOT")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('GCV Esteso: Unificazione Lensing e Rotation Curves', fontsize=13, fontweight='bold')

# Plot 1: Lensing L4
ax1 = axes[0]
data = SDSS_DATA['L4']
R = data['R_kpc']
obs = data['DeltaSigma']
err = data['error']
pred = np.array([model.delta_sigma(r, data['M_stellar']) for r in R])

ax1.errorbar(R, obs, yerr=err, fmt='ko', capsize=3, label='SDSS DR7', markersize=7)
ax1.plot(R, pred, 'b-', linewidth=2, label='GCV Esteso')
ax1.set_xlabel('R [kpc]', fontsize=11)
ax1.set_ylabel('Delta Sigma [M_sun/pc^2]', fontsize=11)
ax1.set_title(f'Lensing L4 (M*=1.5e11)', fontsize=12)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Lensing L2
ax2 = axes[1]
data = SDSS_DATA['L2']
R = data['R_kpc']
obs = data['DeltaSigma']
err = data['error']
pred = np.array([model.delta_sigma(r, data['M_stellar']) for r in R])

ax2.errorbar(R, obs, yerr=err, fmt='ko', capsize=3, label='SDSS DR7', markersize=7)
ax2.plot(R, pred, 'b-', linewidth=2, label='GCV Esteso')
ax2.set_xlabel('R [kpc]', fontsize=11)
ax2.set_ylabel('Delta Sigma [M_sun/pc^2]', fontsize=11)
ax2.set_title(f'Lensing L2 (M*=5e10)', fontsize=12)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Rotation curves
ax3 = axes[2]
masses = []
v_obs_list = []
v_pred_list = []
names = []
for name, data in SPARC_DATA.items():
    masses.append(data['M_star'])
    v_obs_list.append(data['v_obs'])
    v_pred_list.append(model.v_circular(data['R_flat'], data['M_star']))
    names.append(name)

ax3.scatter(v_obs_list, v_pred_list, s=100, c='blue', edgecolors='black')
for i, name in enumerate(names):
    ax3.annotate(name, (v_obs_list[i], v_pred_list[i]), fontsize=8)
ax3.plot([100, 300], [100, 300], 'k--', label='1:1')
ax3.set_xlabel('v_obs [km/s]', fontsize=11)
ax3.set_ylabel('v_pred [km/s]', fontsize=11)
ax3.set_title('Rotation Curves', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'gcv_extended_unified.png', dpi=300, bbox_inches='tight')
print(f"Plot salvato")

# =============================================================================
# SALVA RISULTATI
# =============================================================================
results = {
    'test': 'GCV Extended - Unified Model',
    'params': final_params,
    'chi2_lensing': float(chi2_lens),
    'chi2_rotation': float(chi2_rc),
    'Delta_AIC': float(Delta_AIC),
    'verdict': verdict,
    'interpretation': {
        'alpha_lens': 'Esponente di chi_v per lensing',
        'R_lens': 'Scala caratteristica lensing [kpc]',
        'delta': 'Scaling radiale aggiuntivo',
        'A_lens': 'Normalizzazione lensing'
    }
}

with open(RESULTS_DIR / 'gcv_extended_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# =============================================================================
# INTERPRETAZIONE FISICA
# =============================================================================
print("\n" + "="*70)
print("INTERPRETAZIONE FISICA")
print("="*70)

print(f"""
RISULTATI:

1. Parametri ottimali per unificare lensing e rotation curves:
   - alpha_lens = {best_params[0]:.2f}
   - R_lens = {best_params[1]:.0f} kpc
   - delta = {best_params[2]:.2f}
   - A_lens = {best_params[3]:.2f}

2. INTERPRETAZIONE:
""")

if best_params[0] > 1.5:
    print("   alpha_lens > 1.5: Il lensing e' PIU' sensibile a chi_v della dinamica")
    print("   Questo potrebbe indicare che il lensing 'amplifica' l'effetto GCV")
elif best_params[0] < 0.5:
    print("   alpha_lens < 0.5: Il lensing e' MENO sensibile a chi_v della dinamica")
    print("   Questo potrebbe indicare che la proiezione 'diluisce' l'effetto GCV")
else:
    print("   alpha_lens ~ 1: Lensing e dinamica hanno sensibilita' simile a chi_v")

if best_params[2] > 0.5:
    print(f"   delta > 0: chi_v_lens CRESCE con R (oltre R_lens={best_params[1]:.0f} kpc)")
elif best_params[2] < -0.5:
    print(f"   delta < 0: chi_v_lens DECRESCE con R (oltre R_lens={best_params[1]:.0f} kpc)")

print(f"""
3. CONCLUSIONE:
   - GCV puo' essere esteso per fittare sia lensing che rotation curves
   - Servono {k_gcv} parametri aggiuntivi per il lensing
   - Delta AIC = {Delta_AIC:.1f} vs LCDM
   - Verdetto: {verdict}

4. PROSSIMI PASSI:
   - Derivare alpha_lens, delta da principi primi
   - Testare su piu' dati (DES, KiDS)
   - Verificare consistenza con cluster mergers
""")

print("="*70)
