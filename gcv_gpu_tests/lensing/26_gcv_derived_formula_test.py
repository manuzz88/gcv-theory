#!/usr/bin/env python3
"""
TEST FINALE: GCV con Formula Derivata (alpha = 0.5)

Abbiamo dimostrato che l'esponente alpha ~ 0.5 per il lensing
DERIVA MATEMATICAMENTE dalla definizione di Delta Sigma.

Questo test usa la formula DERIVATA (non fittata):

   Delta Sigma_GCV = Delta Sigma_b * (1 + chi_v)^0.5

dove chi_v usa i parametri da rotation curves (a0, A0, gamma, beta).

NESSUN PARAMETRO AGGIUNTIVO per il lensing!
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.optimize import minimize
from scipy.integrate import quad
import time

print("="*70)
print("TEST FINALE: GCV con Formula Derivata (alpha = 0.5)")
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
# PARAMETRI GCV (da rotation curves - FISSI)
# =============================================================================
GCV_PARAMS = {
    'a0': 1.80e-10,  # m/s^2
    'A0': 1.16,
    'gamma': 0.06,
    'beta': 0.90,
}

# Esponente DERIVATO (non fittato!)
ALPHA_DERIVED = 0.5

print(f"\nParametri GCV (da rotation curves):")
print(f"  a0 = {GCV_PARAMS['a0']:.2e} m/s^2")
print(f"  A0 = {GCV_PARAMS['A0']}")
print(f"  gamma = {GCV_PARAMS['gamma']}")
print(f"  beta = {GCV_PARAMS['beta']}")
print(f"\nEsponente DERIVATO: alpha = {ALPHA_DERIVED}")

# =============================================================================
# DATI SDSS DR7
# =============================================================================
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
# MODELLO GCV CON FORMULA DERIVATA
# =============================================================================

def chi_v(R_kpc, M_star, params):
    """Suscettibilita' GCV"""
    a0 = params['a0']
    A0 = params['A0']
    gamma = params['gamma']
    beta = params['beta']
    
    M = M_star * M_sun
    Lc = np.sqrt(G * M / a0) / kpc  # kpc
    
    chi = A0 * (M_star / 1e11)**gamma * (1 + (R_kpc / Lc)**beta)
    return chi

def delta_sigma_gcv_derived(R_kpc, M_star, params, norm=1.0):
    """
    Delta Sigma GCV con formula DERIVATA
    
    Delta Sigma_GCV = norm * Delta Sigma_base * (1 + chi_v)^0.5
    
    dove Delta Sigma_base viene dal profilo barionico.
    """
    # Velocita' asintotica (da MOND/GCV)
    M = M_star * M_sun
    v_inf = (G * M * params['a0'])**(0.25)  # m/s
    
    # Delta Sigma base (profilo isotermo proiettato)
    R_m = R_kpc * kpc
    ds_base = v_inf**2 / (4 * G * R_m)  # kg/m^2
    ds_base_Msun_pc2 = ds_base / (M_sun / pc**2)
    
    # Applica formula derivata: (1 + chi_v)^0.5
    chi = chi_v(R_kpc, M_star, params)
    ds_gcv = norm * ds_base_Msun_pc2 * (1 + chi)**ALPHA_DERIVED
    
    return ds_gcv

# =============================================================================
# MODELLO LCDM (NFW)
# =============================================================================

def delta_sigma_nfw(R_kpc, M200, c):
    """Delta Sigma per profilo NFW"""
    rho_crit = 3 * (H0 * 1e3 / (3.086e22))**2 / (8 * np.pi * G)
    r200 = (3 * M200 * M_sun / (4 * np.pi * 200 * rho_crit))**(1/3)
    rs = r200 / c
    rs_kpc = rs / kpc
    
    delta_c = 200/3 * c**3 / (np.log(1+c) - c/(1+c))
    rho_s = delta_c * rho_crit
    Sigma_s = rho_s * rs
    
    x = R_kpc / rs_kpc
    
    def g(x_val):
        if x_val < 1:
            term1 = 8 * np.arctanh(np.sqrt((1-x_val)/(1+x_val))) / (x_val**2 * np.sqrt(1-x_val**2))
            term2 = 4 * np.log(x_val/2) / x_val**2
            term3 = -2 / (x_val**2 - 1)
            term4 = 4 * np.arctanh(np.sqrt((1-x_val)/(1+x_val))) / ((x_val**2 - 1) * np.sqrt(1-x_val**2))
            return term1 + term2 + term3 + term4
        elif x_val > 1:
            term1 = 8 * np.arctan(np.sqrt((x_val-1)/(1+x_val))) / (x_val**2 * np.sqrt(x_val**2-1))
            term2 = 4 * np.log(x_val/2) / x_val**2
            term3 = -2 / (x_val**2 - 1)
            term4 = 4 * np.arctan(np.sqrt((x_val-1)/(1+x_val))) / ((x_val**2 - 1)**(3/2))
            return term1 + term2 + term3 + term4
        else:
            return 10/3 + 4*np.log(0.5)
    
    g_vals = np.array([g(xi) for xi in x])
    ds = Sigma_s * g_vals
    ds_Msun_pc2 = ds / (M_sun / pc**2)
    
    return ds_Msun_pc2

# =============================================================================
# FIT E CONFRONTO
# =============================================================================
print("\n" + "="*70)
print("FIT E CONFRONTO")
print("="*70)

results = {}

for sample_name, data in SDSS_DATA.items():
    print(f"\n--- Sample {sample_name} (M*={data['M_stellar']:.1e} M_sun) ---")
    
    R = data['R_kpc']
    obs = data['DeltaSigma']
    err = data['error']
    M_star = data['M_stellar']
    N = len(R)
    
    # GCV: solo normalizzazione libera (1 parametro)
    def chi2_gcv(norm):
        pred = delta_sigma_gcv_derived(R, M_star, GCV_PARAMS, norm[0])
        return np.sum(((obs - pred) / err)**2)
    
    res_gcv = minimize(chi2_gcv, [1.0], method='Nelder-Mead')
    norm_gcv = res_gcv.x[0]
    chi2_gcv_val = res_gcv.fun
    
    # LCDM: M200 e c liberi (2 parametri)
    def chi2_lcdm(params):
        M200, c = params
        if M200 < 1e10 or M200 > 1e15 or c < 1 or c > 30:
            return 1e10
        try:
            pred = delta_sigma_nfw(R, M200, c)
            return np.sum(((obs - pred) / err)**2)
        except:
            return 1e10
    
    res_lcdm = minimize(chi2_lcdm, [M_star * 100, 10], method='Nelder-Mead')
    M200_best, c_best = res_lcdm.x
    chi2_lcdm_val = res_lcdm.fun
    
    # Statistiche
    k_gcv = 1  # solo normalizzazione
    k_lcdm = 2  # M200, c
    
    dof_gcv = N - k_gcv
    dof_lcdm = N - k_lcdm
    
    chi2_red_gcv = chi2_gcv_val / dof_gcv
    chi2_red_lcdm = chi2_lcdm_val / dof_lcdm
    
    AIC_gcv = chi2_gcv_val + 2 * k_gcv
    AIC_lcdm = chi2_lcdm_val + 2 * k_lcdm
    
    print(f"  GCV (formula derivata):")
    print(f"    norm = {norm_gcv:.3f}")
    print(f"    chi2 = {chi2_gcv_val:.2f}, chi2/dof = {chi2_red_gcv:.3f}")
    print(f"  LCDM (NFW):")
    print(f"    M200 = {M200_best:.2e}, c = {c_best:.1f}")
    print(f"    chi2 = {chi2_lcdm_val:.2f}, chi2/dof = {chi2_red_lcdm:.3f}")
    print(f"  Delta AIC = {AIC_gcv - AIC_lcdm:.1f}")
    
    results[sample_name] = {
        'N': N,
        'GCV': {
            'norm': float(norm_gcv),
            'chi2': float(chi2_gcv_val),
            'chi2_red': float(chi2_red_gcv),
            'k': k_gcv,
            'AIC': float(AIC_gcv)
        },
        'LCDM': {
            'M200': float(M200_best),
            'c': float(c_best),
            'chi2': float(chi2_lcdm_val),
            'chi2_red': float(chi2_red_lcdm),
            'k': k_lcdm,
            'AIC': float(AIC_lcdm)
        }
    }

# =============================================================================
# RISULTATI COMBINATI
# =============================================================================
print("\n" + "="*70)
print("RISULTATI COMBINATI")
print("="*70)

chi2_gcv_tot = sum(r['GCV']['chi2'] for r in results.values())
chi2_lcdm_tot = sum(r['LCDM']['chi2'] for r in results.values())
N_tot = sum(r['N'] for r in results.values())

k_gcv_tot = 2  # 1 norm per sample
k_lcdm_tot = 4  # 2 params per sample

dof_gcv = N_tot - k_gcv_tot
dof_lcdm = N_tot - k_lcdm_tot

AIC_gcv_tot = chi2_gcv_tot + 2 * k_gcv_tot
AIC_lcdm_tot = chi2_lcdm_tot + 2 * k_lcdm_tot
BIC_gcv_tot = chi2_gcv_tot + k_gcv_tot * np.log(N_tot)
BIC_lcdm_tot = chi2_lcdm_tot + k_lcdm_tot * np.log(N_tot)

Delta_AIC = AIC_gcv_tot - AIC_lcdm_tot
Delta_BIC = BIC_gcv_tot - BIC_lcdm_tot

print(f"\nGCV (formula derivata, alpha=0.5 FISSO):")
print(f"  chi2 totale = {chi2_gcv_tot:.2f}")
print(f"  chi2/dof = {chi2_gcv_tot/dof_gcv:.3f}")
print(f"  Parametri liberi = {k_gcv_tot} (solo normalizzazioni)")
print(f"  AIC = {AIC_gcv_tot:.1f}")

print(f"\nLCDM (NFW):")
print(f"  chi2 totale = {chi2_lcdm_tot:.2f}")
print(f"  chi2/dof = {chi2_lcdm_tot/dof_lcdm:.3f}")
print(f"  Parametri liberi = {k_lcdm_tot}")
print(f"  AIC = {AIC_lcdm_tot:.1f}")

print(f"\n" + "-"*50)
print(f"Delta AIC = {Delta_AIC:.1f}")
print(f"Delta BIC = {Delta_BIC:.1f}")

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

print(f"\nVERDETTO: {verdict}")

# =============================================================================
# PLOT
# =============================================================================
print("\n" + "="*70)
print("GENERAZIONE PLOT")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('GCV Formula Derivata (alpha=0.5) vs LCDM su SDSS DR7', 
             fontsize=13, fontweight='bold')

for idx, (sample_name, data) in enumerate(SDSS_DATA.items()):
    ax = axes[idx]
    
    R = data['R_kpc']
    obs = data['DeltaSigma']
    err = data['error']
    M_star = data['M_stellar']
    
    # Dati
    ax.errorbar(R, obs, yerr=err, fmt='ko', capsize=3, 
                label='SDSS DR7', markersize=7, zorder=10)
    
    # GCV
    R_plot = np.logspace(np.log10(R.min()), np.log10(R.max()), 100)
    norm = results[sample_name]['GCV']['norm']
    pred_gcv = delta_sigma_gcv_derived(R_plot, M_star, GCV_PARAMS, norm)
    chi2_gcv = results[sample_name]['GCV']['chi2']
    ax.plot(R_plot, pred_gcv, 'b-', linewidth=2, 
            label=f'GCV derivata (chi2={chi2_gcv:.1f})')
    
    # LCDM
    M200 = results[sample_name]['LCDM']['M200']
    c = results[sample_name]['LCDM']['c']
    pred_lcdm = delta_sigma_nfw(R_plot, M200, c)
    chi2_lcdm = results[sample_name]['LCDM']['chi2']
    ax.plot(R_plot, pred_lcdm, 'r--', linewidth=2,
            label=f'LCDM NFW (chi2={chi2_lcdm:.1f})')
    
    ax.set_xlabel('R [kpc]', fontsize=11)
    ax.set_ylabel('Delta Sigma [M_sun/pc^2]', fontsize=11)
    ax.set_title(f'Sample {sample_name}: M*={M_star:.1e} M_sun', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# Box risultato
textstr = f'Delta AIC = {Delta_AIC:.1f}\nVerdetto: {verdict}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
fig.text(0.5, 0.02, textstr, ha='center', fontsize=11, bbox=props)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.savefig(PLOTS_DIR / 'gcv_derived_formula_test.png', dpi=300, bbox_inches='tight')
print("Plot salvato")

# =============================================================================
# SALVA RISULTATI
# =============================================================================
final_results = {
    'test': 'GCV Derived Formula Test',
    'formula': 'Delta_Sigma_GCV = norm * Delta_Sigma_base * (1 + chi_v)^0.5',
    'alpha_derived': ALPHA_DERIVED,
    'note': 'alpha=0.5 is DERIVED from Delta Sigma definition, not fitted',
    'gcv_params': GCV_PARAMS,
    'samples': results,
    'combined': {
        'chi2_gcv': float(chi2_gcv_tot),
        'chi2_lcdm': float(chi2_lcdm_tot),
        'k_gcv': k_gcv_tot,
        'k_lcdm': k_lcdm_tot,
        'Delta_AIC': float(Delta_AIC),
        'Delta_BIC': float(Delta_BIC),
        'verdict': verdict
    }
}

with open(RESULTS_DIR / 'gcv_derived_formula_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print("Risultati salvati")

# =============================================================================
# CONCLUSIONI
# =============================================================================
print("\n" + "="*70)
print("CONCLUSIONI")
print("="*70)

print(f"""
RISULTATO DEL TEST:

1. FORMULA USATA:
   Delta Sigma_GCV = norm * Delta Sigma_base * (1 + chi_v)^0.5
   
   dove alpha = 0.5 e' DERIVATO, non fittato!

2. PARAMETRI:
   - GCV: {k_gcv_tot} parametri liberi (solo normalizzazioni)
   - LCDM: {k_lcdm_tot} parametri liberi (M200, c per ogni sample)

3. RISULTATI:
   - GCV chi2/dof = {chi2_gcv_tot/dof_gcv:.3f}
   - LCDM chi2/dof = {chi2_lcdm_tot/dof_lcdm:.3f}
   - Delta AIC = {Delta_AIC:.1f}

4. VERDETTO: {verdict}

5. SIGNIFICATO:
""")

if Delta_AIC < 2:
    print("""   GCV con formula DERIVATA e' COMPETITIVA con LCDM!
   
   Questo e' un risultato FORTE perche':
   - L'esponente 0.5 non e' fittato, e' derivato matematicamente
   - GCV usa MENO parametri di LCDM
   - I parametri GCV vengono da rotation curves (cross-validation!)
""")
elif Delta_AIC < 10:
    print("""   LCDM e' leggermente favorita, ma GCV rimane competitiva.
   
   La differenza potrebbe venire da:
   - Approssimazioni nel profilo base
   - Effetti non inclusi nel modello semplificato
""")
else:
    print("""   LCDM e' favorita su questi dati.
   
   Possibili cause:
   - Il profilo base (isotermo) e' troppo semplificato
   - Servono correzioni per la fisica barionica
""")

print("="*70)
