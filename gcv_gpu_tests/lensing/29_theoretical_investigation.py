#!/usr/bin/env python3
"""
INVESTIGAZIONE TEORICA: Perche' GCV non fitta il lensing?

PROBLEMA:
- GCV con A0=1.16 funziona per rotation curves
- Ma per lensing serve A0 molto diverso
- Perche'?

IPOTESI DA INVESTIGARE:

1. DIFFERENZA TRA DINAMICA E LENSING
   - Dinamica: misura la FORZA (dPhi/dr)
   - Lensing: misura il POTENZIALE (Phi)
   - Forse chi_v agisce diversamente su forza vs potenziale?

2. SCALA SPAZIALE
   - Rotation curves: R ~ 1-30 kpc
   - Lensing SDSS: R ~ 40-2600 kpc
   - Forse chi_v ha comportamento diverso a scale diverse?

3. PROIEZIONE
   - Rotation curves: misura nel piano del disco
   - Lensing: integra lungo la linea di vista
   - La proiezione potrebbe modificare l'effetto

4. FORMA FUNZIONALE
   - Forse chi_v per lensing ha forma diversa
   - Non (1 + chi_v)^0.5 ma qualcos'altro
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.optimize import minimize, curve_fit
from scipy.integrate import quad

print("="*70)
print("INVESTIGAZIONE TEORICA: Lensing vs Rotation Curves")
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

# Rotation curves (SPARC)
SPARC_DATA = {
    'NGC3198': {'M_star': 3.5e10, 'R_kpc': 15, 'v_obs': 150},
    'NGC2403': {'M_star': 5e9, 'R_kpc': 10, 'v_obs': 130},
    'NGC7331': {'M_star': 1e11, 'R_kpc': 20, 'v_obs': 250},
    'NGC5055': {'M_star': 6e10, 'R_kpc': 25, 'v_obs': 200},
    'NGC6946': {'M_star': 4e10, 'R_kpc': 15, 'v_obs': 180},
}

# Lensing (SDSS)
SDSS_DATA = {
    'L4': {
        'M_stellar': 1.5e11,
        'R_kpc': np.array([42, 65, 104, 165, 261, 414, 656, 1039, 1645, 2607]),
        'DeltaSigma': np.array([127.3, 98.2, 74.1, 55.8, 40.2, 27.1, 17.8, 11.2, 7.1, 4.3]),
        'error': np.array([11.2, 8.5, 6.4, 4.8, 3.5, 2.4, 1.6, 1.1, 0.8, 0.6]),
    },
    'L2': {
        'M_stellar': 5e10,
        'R_kpc': np.array([42, 65, 104, 165, 261, 414, 656, 1039, 1645]),
        'DeltaSigma': np.array([68.5, 54.2, 41.3, 31.8, 23.1, 15.7, 10.2, 6.4, 4.0]),
        'error': np.array([7.8, 5.9, 4.4, 3.3, 2.4, 1.7, 1.2, 0.9, 0.7]),
    }
}

# Parametri GCV base
a0 = 1.80e-10

# =============================================================================
# IPOTESI 1: SCALA SPAZIALE
# =============================================================================
print("\n" + "="*70)
print("IPOTESI 1: DIPENDENZA DALLA SCALA SPAZIALE")
print("="*70)

print("""
Le rotation curves misurano a R ~ 1-30 kpc
Il lensing SDSS misura a R ~ 40-2600 kpc

Forse chi_v ha un CAMBIO DI REGIME a scale diverse?

Proposta: chi_v con transizione

chi_v(R) = A0 * (M/M0)^gamma * [1 + (R/Lc)^beta] * f_transition(R)

dove f_transition(R) = 1 / (1 + (R/R_trans)^n)

Questo ridurrebbe chi_v a grandi R, spiegando perche' il lensing
vede meno "boost" delle rotation curves.
""")

def chi_v_with_transition(R_kpc, M_star, A0, beta, R_trans, n_trans):
    """chi_v con transizione a grandi scale"""
    M = M_star * M_sun
    Lc = np.sqrt(G * M / a0) / kpc
    
    chi_base = A0 * (M_star / 1e11)**0.06 * (1 + (R_kpc / Lc)**beta)
    
    # Transizione: chi_v si "spegne" a R > R_trans
    f_trans = 1 / (1 + (R_kpc / R_trans)**n_trans)
    
    return chi_base * f_trans

# Test: quale R_trans serve?
print("\nCercando R_trans ottimale...")

def chi2_transition(params):
    A0, R_trans, n_trans = params
    if A0 < 0.1 or A0 > 10 or R_trans < 10 or R_trans > 1000 or n_trans < 0.1 or n_trans > 5:
        return 1e10
    
    chi2 = 0
    
    # Lensing
    for name, data in SDSS_DATA.items():
        R = data['R_kpc']
        obs = data['DeltaSigma']
        err = data['error']
        M_star = data['M_stellar']
        
        # Delta Sigma con transizione
        M = M_star * M_sun
        v_inf = (G * M * a0)**(0.25)
        R_m = R * kpc
        ds_base = v_inf**2 / (4 * G * R_m) / (M_sun / pc**2)
        
        chi_v = chi_v_with_transition(R, M_star, A0, 0.9, R_trans, n_trans)
        pred = ds_base * (1 + chi_v)**0.5
        
        chi2 += np.sum(((obs - pred) / err)**2)
    
    # Rotation curves (penalita' se non fittano)
    for name, data in SPARC_DATA.items():
        M_star = data['M_star']
        R = data['R_kpc']
        v_obs = data['v_obs']
        
        M = M_star * M_sun
        v_inf = (G * M * a0)**(0.25) / 1000  # km/s
        
        chi_v = chi_v_with_transition(R, M_star, A0, 0.9, R_trans, n_trans)
        v_pred = v_inf * (1 + chi_v)**0.25  # approssimazione
        
        chi2 += ((v_obs - v_pred) / (0.1 * v_obs))**2
    
    return chi2

from scipy.optimize import differential_evolution

bounds = [(0.5, 5), (20, 500), (0.5, 3)]
result = differential_evolution(chi2_transition, bounds, seed=42, maxiter=100)
A0_opt, R_trans_opt, n_trans_opt = result.x

print(f"\nParametri ottimali con transizione:")
print(f"  A0 = {A0_opt:.2f}")
print(f"  R_trans = {R_trans_opt:.0f} kpc")
print(f"  n_trans = {n_trans_opt:.2f}")
print(f"  Chi2 = {result.fun:.2f}")

# =============================================================================
# IPOTESI 2: LENSING VEDE MASSA DIVERSA
# =============================================================================
print("\n" + "="*70)
print("IPOTESI 2: LENSING VEDE MASSA DIVERSA")
print("="*70)

print("""
In GCV, la massa DINAMICA e':
   M_dyn = M_b * (1 + chi_v)

Ma il lensing potrebbe vedere una massa DIVERSA.

In relativita' generale, il lensing dipende dalla somma:
   Phi + Psi (potenziali scalari)

In GCV modificata, potremmo avere:
   Phi_lens = Phi_Newton * (1 + chi_v)^alpha_Phi
   
con alpha_Phi != 1

Questo spiegherebbe la differenza!
""")

def test_alpha_phi(alpha_Phi):
    """Testa diversi valori di alpha_Phi"""
    chi2 = 0
    
    for name, data in SDSS_DATA.items():
        R = data['R_kpc']
        obs = data['DeltaSigma']
        err = data['error']
        M_star = data['M_stellar']
        
        M = M_star * M_sun
        Lc = np.sqrt(G * M / a0) / kpc
        v_inf = (G * M * a0)**(0.25)
        
        R_m = R * kpc
        ds_base = v_inf**2 / (4 * G * R_m) / (M_sun / pc**2)
        
        chi_v = 1.16 * (M_star / 1e11)**0.06 * (1 + (R / Lc)**0.9)
        
        # Lensing con alpha_Phi
        pred = ds_base * (1 + chi_v)**alpha_Phi
        
        chi2 += np.sum(((obs - pred) / err)**2)
    
    return chi2

print("\nTestando diversi alpha_Phi:")
print(f"{'alpha_Phi':>10} {'Chi2':>10} {'Chi2/dof':>10}")
print("-" * 35)

best_alpha = 0
best_chi2 = 1e10
N_tot = sum(len(d['R_kpc']) for d in SDSS_DATA.values())

for alpha in np.linspace(0.1, 1.0, 10):
    chi2 = test_alpha_phi(alpha)
    chi2_red = chi2 / (N_tot - 1)
    print(f"{alpha:>10.2f} {chi2:>10.1f} {chi2_red:>10.2f}")
    if chi2 < best_chi2:
        best_chi2 = chi2
        best_alpha = alpha

print(f"\nMigliore alpha_Phi = {best_alpha:.2f}")

# =============================================================================
# IPOTESI 3: FORMA FUNZIONALE DIVERSA
# =============================================================================
print("\n" + "="*70)
print("IPOTESI 3: FORMA FUNZIONALE DIVERSA PER LENSING")
print("="*70)

print("""
Forse il lensing richiede una forma DIVERSA di chi_v.

Proposta: chi_v_lens dipende dal GRADIENTE di chi_v, non da chi_v stesso.

Motivazione fisica:
- Il lensing e' sensibile alla CURVATURA dello spaziotempo
- La curvatura dipende dalle DERIVATE SECONDE del potenziale
- Quindi chi_v_lens ~ d^2(chi_v)/dR^2

Proviamo:
chi_v_lens = A_lens * |d(chi_v)/dR| * R
""")

def chi_v_gradient_based(R_kpc, M_star, A_lens):
    """chi_v basato sul gradiente"""
    M = M_star * M_sun
    Lc = np.sqrt(G * M / a0) / kpc
    beta = 0.9
    A0 = 1.16
    gamma = 0.06
    
    # chi_v standard
    chi_v = A0 * (M_star / 1e11)**gamma * (1 + (R_kpc / Lc)**beta)
    
    # Gradiente: d(chi_v)/dR
    d_chi_dR = A0 * (M_star / 1e11)**gamma * beta * (R_kpc / Lc)**(beta - 1) / Lc
    
    # chi_v_lens basato sul gradiente
    chi_v_lens = A_lens * np.abs(d_chi_dR) * R_kpc
    
    return chi_v_lens

def test_gradient_model(A_lens):
    chi2 = 0
    
    for name, data in SDSS_DATA.items():
        R = data['R_kpc']
        obs = data['DeltaSigma']
        err = data['error']
        M_star = data['M_stellar']
        
        M = M_star * M_sun
        v_inf = (G * M * a0)**(0.25)
        R_m = R * kpc
        ds_base = v_inf**2 / (4 * G * R_m) / (M_sun / pc**2)
        
        chi_v_lens = chi_v_gradient_based(R, M_star, A_lens)
        pred = ds_base * (1 + chi_v_lens)**0.5
        
        chi2 += np.sum(((obs - pred) / err)**2)
    
    return chi2

res_grad = minimize(lambda x: test_gradient_model(x[0]), [1.0], method='Nelder-Mead')
A_lens_grad = res_grad.x[0]
chi2_grad = res_grad.fun

print(f"\nModello basato sul gradiente:")
print(f"  A_lens = {A_lens_grad:.2f}")
print(f"  Chi2 = {chi2_grad:.2f}")
print(f"  Chi2/dof = {chi2_grad/(N_tot-1):.2f}")

# =============================================================================
# IPOTESI 4: CONTRIBUTO AMBIENTALE
# =============================================================================
print("\n" + "="*70)
print("IPOTESI 4: CONTRIBUTO AMBIENTALE (CLUSTERING)")
print("="*70)

print("""
Le galassie SDSS sono in ambienti densi (gruppi, cluster).
Il lensing potrebbe includere contributo da:
- Galassie vicine
- Alone del gruppo/cluster

Proposta: aggiungere termine di "ambiente"

Delta Sigma = Delta Sigma_GCV + Delta Sigma_env

dove Delta Sigma_env ~ costante o ~ R^(-1)
""")

def model_with_environment(params, R, M_star):
    """Modello GCV + contributo ambientale"""
    A0, DS_env = params
    
    M = M_star * M_sun
    Lc = np.sqrt(G * M / a0) / kpc
    v_inf = (G * M * a0)**(0.25)
    
    R_m = R * kpc
    ds_base = v_inf**2 / (4 * G * R_m) / (M_sun / pc**2)
    
    chi_v = A0 * (M_star / 1e11)**0.06 * (1 + (R / Lc)**0.9)
    ds_gcv = ds_base * (1 + chi_v)**0.5
    
    # Contributo ambientale (costante)
    ds_total = ds_gcv + DS_env
    
    return ds_total

def chi2_environment(params):
    A0, DS_env_L4, DS_env_L2 = params
    if A0 < 0.1 or DS_env_L4 < 0 or DS_env_L2 < 0:
        return 1e10
    
    chi2 = 0
    
    for name, data in SDSS_DATA.items():
        R = data['R_kpc']
        obs = data['DeltaSigma']
        err = data['error']
        M_star = data['M_stellar']
        
        DS_env = DS_env_L4 if name == 'L4' else DS_env_L2
        pred = model_with_environment([A0, DS_env], R, M_star)
        
        chi2 += np.sum(((obs - pred) / err)**2)
    
    return chi2

res_env = minimize(chi2_environment, [1.16, 10, 5], method='Nelder-Mead')
A0_env, DS_env_L4, DS_env_L2 = res_env.x
chi2_env = res_env.fun

print(f"\nModello GCV + ambiente:")
print(f"  A0 = {A0_env:.2f}")
print(f"  DS_env (L4) = {DS_env_L4:.1f} M_sun/pc^2")
print(f"  DS_env (L2) = {DS_env_L2:.1f} M_sun/pc^2")
print(f"  Chi2 = {chi2_env:.2f}")
print(f"  Chi2/dof = {chi2_env/(N_tot-3):.2f}")

# =============================================================================
# CONFRONTO MODELLI
# =============================================================================
print("\n" + "="*70)
print("CONFRONTO MODELLI")
print("="*70)

chi2_lcdm = 27.79

models = {
    'LCDM (NFW)': {'chi2': chi2_lcdm, 'k': 4},
    'GCV originale': {'chi2': 151.99, 'k': 2},
    'GCV + transizione': {'chi2': result.fun, 'k': 4},
    'GCV alpha_Phi': {'chi2': best_chi2, 'k': 2},
    'GCV gradiente': {'chi2': chi2_grad, 'k': 2},
    'GCV + ambiente': {'chi2': chi2_env, 'k': 4},
}

print(f"\n{'Modello':<20} {'Chi2':>10} {'k':>5} {'AIC':>10} {'Delta AIC':>12}")
print("-" * 60)

AIC_lcdm = chi2_lcdm + 2 * 4

for name, m in models.items():
    AIC = m['chi2'] + 2 * m['k']
    delta = AIC - AIC_lcdm
    print(f"{name:<20} {m['chi2']:>10.1f} {m['k']:>5} {AIC:>10.1f} {delta:>12.1f}")

# =============================================================================
# MIGLIORE MODELLO
# =============================================================================
print("\n" + "="*70)
print("ANALISI DEL MIGLIORE MODELLO GCV")
print("="*70)

# Trova il migliore
best_model = min(models.items(), key=lambda x: x[1]['chi2'] + 2*x[1]['k'] if 'GCV' in x[0] else 1e10)
print(f"\nMigliore modello GCV: {best_model[0]}")
print(f"  Chi2 = {best_model[1]['chi2']:.1f}")
print(f"  AIC = {best_model[1]['chi2'] + 2*best_model[1]['k']:.1f}")

# =============================================================================
# PLOT
# =============================================================================
print("\n" + "="*70)
print("GENERAZIONE PLOT")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Investigazione Teorica: Modelli GCV per Lensing', fontsize=14, fontweight='bold')

# Plot 1: Confronto modelli su L4
ax1 = axes[0, 0]
data = SDSS_DATA['L4']
R = data['R_kpc']
obs = data['DeltaSigma']
err = data['error']
M_star = data['M_stellar']

ax1.errorbar(R, obs, yerr=err, fmt='ko', capsize=3, label='SDSS L4', markersize=7)

R_plot = np.logspace(np.log10(R.min()), np.log10(R.max()), 100)

# GCV originale
M = M_star * M_sun
Lc = np.sqrt(G * M / a0) / kpc
v_inf = (G * M * a0)**(0.25)
ds_base = v_inf**2 / (4 * G * R_plot * kpc) / (M_sun / pc**2)
chi_v_orig = 1.16 * (M_star / 1e11)**0.06 * (1 + (R_plot / Lc)**0.9)
pred_orig = ds_base * (1 + chi_v_orig)**0.5
ax1.plot(R_plot, pred_orig, 'b--', alpha=0.5, label='GCV originale')

# GCV + transizione
chi_v_trans = chi_v_with_transition(R_plot, M_star, A0_opt, 0.9, R_trans_opt, n_trans_opt)
pred_trans = ds_base * (1 + chi_v_trans)**0.5
ax1.plot(R_plot, pred_trans, 'g-', linewidth=2, label=f'GCV + trans (R_t={R_trans_opt:.0f})')

# GCV + ambiente
pred_env = model_with_environment([A0_env, DS_env_L4], R_plot, M_star)
ax1.plot(R_plot, pred_env, 'r-', linewidth=2, label=f'GCV + ambiente')

ax1.set_xlabel('R [kpc]')
ax1.set_ylabel('Delta Sigma [M_sun/pc^2]')
ax1.set_title('Sample L4')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: chi_v vs R
ax2 = axes[0, 1]
R_test = np.logspace(0, 3.5, 100)
chi_v_std = 1.16 * (1e11 / 1e11)**0.06 * (1 + (R_test / 10)**0.9)
chi_v_tr = chi_v_with_transition(R_test, 1e11, A0_opt, 0.9, R_trans_opt, n_trans_opt)

ax2.loglog(R_test, chi_v_std, 'b-', label='chi_v standard')
ax2.loglog(R_test, chi_v_tr, 'g-', label=f'chi_v con transizione')
ax2.axvline(R_trans_opt, color='r', linestyle='--', label=f'R_trans={R_trans_opt:.0f} kpc')
ax2.axvspan(1, 30, alpha=0.2, color='blue', label='Range rotation curves')
ax2.axvspan(40, 2600, alpha=0.2, color='green', label='Range lensing')

ax2.set_xlabel('R [kpc]')
ax2.set_ylabel('chi_v')
ax2.set_title('Suscettibilita vs Raggio')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: Residui
ax3 = axes[1, 0]
for name, data in SDSS_DATA.items():
    R = data['R_kpc']
    obs = data['DeltaSigma']
    err = data['error']
    M_star = data['M_stellar']
    
    pred = model_with_environment([A0_env, DS_env_L4 if name=='L4' else DS_env_L2], R, M_star)
    residuals = (obs - pred) / err
    
    ax3.scatter(R, residuals, label=name, s=50)

ax3.axhline(0, color='k', linestyle='-')
ax3.axhline(2, color='r', linestyle='--', alpha=0.5)
ax3.axhline(-2, color='r', linestyle='--', alpha=0.5)
ax3.set_xlabel('R [kpc]')
ax3.set_ylabel('Residui / Errore')
ax3.set_title('Residui (GCV + ambiente)')
ax3.set_xscale('log')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Confronto AIC
ax4 = axes[1, 1]
model_names = list(models.keys())
AICs = [m['chi2'] + 2*m['k'] for m in models.values()]
colors = ['red' if 'LCDM' in n else 'blue' for n in model_names]

bars = ax4.barh(model_names, AICs, color=colors, alpha=0.7)
ax4.axvline(AIC_lcdm, color='red', linestyle='--', label='LCDM')
ax4.set_xlabel('AIC')
ax4.set_title('Confronto Modelli (AIC)')
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'theoretical_investigation.png', dpi=300, bbox_inches='tight')
print("Plot salvato")

# =============================================================================
# CONCLUSIONI
# =============================================================================
print("\n" + "="*70)
print("CONCLUSIONI")
print("="*70)

print(f"""
RISULTATI DELL'INVESTIGAZIONE:

1. GCV + TRANSIZIONE (R_trans = {R_trans_opt:.0f} kpc):
   - chi_v si "spegne" gradualmente a grandi scale
   - Chi2 = {result.fun:.1f}
   - Interpretazione: vacuum coherence ha scala massima

2. GCV + AMBIENTE:
   - Aggiunge contributo costante da ambiente
   - Chi2 = {chi2_env:.1f}
   - Interpretazione: galassie SDSS sono in gruppi/cluster

3. GCV ALPHA_PHI:
   - Lensing vede (1 + chi_v)^{best_alpha:.2f} invece di ^0.5
   - Chi2 = {best_chi2:.1f}
   - Interpretazione: relazione potenziale-lensing diversa

MIGLIORE MODELLO GCV: {best_model[0]}

CONFRONTO CON LCDM:
- LCDM AIC = {AIC_lcdm:.1f}
- Migliore GCV AIC = {best_model[1]['chi2'] + 2*best_model[1]['k']:.1f}
- Delta AIC = {best_model[1]['chi2'] + 2*best_model[1]['k'] - AIC_lcdm:.1f}
""")

# Salva risultati
results = {
    'investigation': 'Theoretical Investigation',
    'models_tested': {
        'transition': {
            'A0': float(A0_opt),
            'R_trans': float(R_trans_opt),
            'n_trans': float(n_trans_opt),
            'chi2': float(result.fun)
        },
        'alpha_phi': {
            'best_alpha': float(best_alpha),
            'chi2': float(best_chi2)
        },
        'gradient': {
            'A_lens': float(A_lens_grad),
            'chi2': float(chi2_grad)
        },
        'environment': {
            'A0': float(A0_env),
            'DS_env_L4': float(DS_env_L4),
            'DS_env_L2': float(DS_env_L2),
            'chi2': float(chi2_env)
        }
    },
    'best_gcv_model': best_model[0],
    'comparison_with_lcdm': {
        'lcdm_aic': float(AIC_lcdm),
        'best_gcv_aic': float(best_model[1]['chi2'] + 2*best_model[1]['k']),
        'delta_aic': float(best_model[1]['chi2'] + 2*best_model[1]['k'] - AIC_lcdm)
    }
}

with open(RESULTS_DIR / 'theoretical_investigation.json', 'w') as f:
    json.dump(results, f, indent=2)

print("="*70)
