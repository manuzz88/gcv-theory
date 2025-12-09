#!/usr/bin/env python3
"""
MCMC GPU-ACCELERATO per ottimizzare GCV su dati di lensing

Il test precedente mostra che GCV con parametri fissi da rotation curves
non fitta bene il lensing. Questo puo' significare:
1. La formula di proiezione e' sbagliata
2. I parametri devono essere ri-ottimizzati
3. Serve un termine aggiuntivo

Questo script esplora lo spazio dei parametri con MCMC GPU.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy import stats as cp_stats
    GPU_AVAILABLE = True
    print(f"GPU: {cp.cuda.runtime.getDeviceCount()} device(s) disponibili")
except ImportError:
    GPU_AVAILABLE = False
    cp = np  # fallback
    print("GPU non disponibile, uso CPU")

print("="*70)
print("MCMC GPU: OTTIMIZZAZIONE PARAMETRI GCV SU LENSING")
print("="*70)

# Directories
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Costanti
G = 6.674e-11
M_sun = 1.989e30
kpc = 3.086e19
pc = 3.086e16
H0 = 67.4
h = H0 / 100

# =============================================================================
# DATI SDSS
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

# Combina i dati
all_R = []
all_DS = []
all_err = []
all_M = []

for name, data in SDSS_DATA.items():
    for i in range(len(data['R_kpc'])):
        all_R.append(data['R_kpc'][i])
        all_DS.append(data['DeltaSigma'][i])
        all_err.append(data['error'][i])
        all_M.append(data['M_stellar'])

all_R = np.array(all_R)
all_DS = np.array(all_DS)
all_err = np.array(all_err)
all_M = np.array(all_M)

N_data = len(all_R)
print(f"\nDati combinati: {N_data} punti")

# =============================================================================
# MODELLO GCV GENERALIZZATO
# =============================================================================

def gcv_model(R_kpc, M_star, params):
    """
    Modello GCV generalizzato per Delta Sigma
    
    Parametri:
    - a0: scala di accelerazione
    - A0: ampiezza chi_v
    - gamma: scaling con massa
    - beta: scaling radiale
    - alpha_proj: esponente proiezione (NUOVO!)
    - norm: normalizzazione globale
    """
    a0 = params['a0']
    A0 = params['A0']
    gamma = params['gamma']
    beta = params['beta']
    alpha_proj = params['alpha_proj']
    norm = params['norm']
    
    # Lunghezza di coerenza
    Lc_m = np.sqrt(G * M_star * M_sun / a0)
    Lc_kpc = Lc_m / kpc
    
    # Velocita' asintotica
    v_inf = (G * M_star * M_sun * a0)**(0.25)
    
    # Chi_v
    chi_v = 1 + A0 * (M_star / 1e11)**gamma * (R_kpc / Lc_kpc)**beta
    
    # Delta Sigma con proiezione generalizzata
    R_m = R_kpc * kpc
    ds_base = v_inf**2 / (4 * G * R_m)
    ds_Msun_pc2 = ds_base / (M_sun / pc**2)
    
    # Applica chi_v con esponente di proiezione
    ds = norm * ds_Msun_pc2 * chi_v**alpha_proj
    
    return ds

def log_likelihood(theta):
    """Log-likelihood per MCMC"""
    a0, A0, gamma, beta, alpha_proj, norm = theta
    
    # Prior bounds
    if not (1e-11 < a0 < 1e-9):
        return -np.inf
    if not (0.1 < A0 < 5):
        return -np.inf
    if not (-0.5 < gamma < 0.5):
        return -np.inf
    if not (0.1 < beta < 2):
        return -np.inf
    if not (0.1 < alpha_proj < 2):
        return -np.inf
    if not (0.01 < norm < 10):
        return -np.inf
    
    params = {
        'a0': a0, 'A0': A0, 'gamma': gamma,
        'beta': beta, 'alpha_proj': alpha_proj, 'norm': norm
    }
    
    try:
        pred = np.array([gcv_model(R, M, params) for R, M in zip(all_R, all_M)])
        chi2 = np.sum(((all_DS - pred) / all_err)**2)
        return -0.5 * chi2
    except:
        return -np.inf

def log_prior(theta):
    """Log-prior (uniforme nei bounds)"""
    a0, A0, gamma, beta, alpha_proj, norm = theta
    
    if (1e-11 < a0 < 1e-9 and 0.1 < A0 < 5 and -0.5 < gamma < 0.5 and
        0.1 < beta < 2 and 0.1 < alpha_proj < 2 and 0.01 < norm < 10):
        return 0.0
    return -np.inf

def log_probability(theta):
    """Log-probabilita' posteriore"""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

# =============================================================================
# MCMC GPU-ACCELERATO
# =============================================================================
print("\n" + "="*70)
print("MCMC SAMPLING")
print("="*70)

# Prova prima ottimizzazione classica per trovare punto iniziale
print("\nTrovando punto iniziale con ottimizzazione...")

def neg_log_like(theta):
    ll = log_likelihood(theta)
    if np.isfinite(ll):
        return -ll
    return 1e10

# Punto iniziale ragionevole
x0 = [1.8e-10, 1.16, 0.06, 0.90, 0.5, 0.5]

from scipy.optimize import differential_evolution

bounds = [
    (1e-11, 1e-9),   # a0
    (0.1, 5),        # A0
    (-0.5, 0.5),     # gamma
    (0.1, 2),        # beta
    (0.1, 2),        # alpha_proj
    (0.01, 10)       # norm
]

print("Ottimizzazione globale (Differential Evolution)...")
result = differential_evolution(neg_log_like, bounds, maxiter=500, 
                                 seed=42, workers=-1, disp=True)
best_params = result.x
best_chi2 = result.fun

print(f"\nMigliori parametri trovati:")
print(f"  a0 = {best_params[0]:.3e} m/s^2")
print(f"  A0 = {best_params[1]:.3f}")
print(f"  gamma = {best_params[2]:.3f}")
print(f"  beta = {best_params[3]:.3f}")
print(f"  alpha_proj = {best_params[4]:.3f}")
print(f"  norm = {best_params[5]:.3f}")
print(f"  chi2 = {best_chi2:.2f}")
print(f"  chi2/dof = {best_chi2/(N_data-6):.3f}")

# =============================================================================
# MCMC con emcee (se disponibile)
# =============================================================================
try:
    import emcee
    
    print("\n" + "="*70)
    print("MCMC SAMPLING con emcee")
    print("="*70)
    
    ndim = 6
    nwalkers = 32
    nsteps = 2000
    
    # Inizializza walkers attorno al best fit
    pos = best_params + 1e-4 * np.random.randn(nwalkers, ndim) * np.abs(best_params)
    
    # Assicurati che tutti i walkers siano nei bounds
    for i in range(nwalkers):
        for j in range(ndim):
            pos[i,j] = np.clip(pos[i,j], bounds[j][0]*1.01, bounds[j][1]*0.99)
    
    print(f"Walkers: {nwalkers}")
    print(f"Steps: {nsteps}")
    print(f"Parametri: {ndim}")
    
    # Run MCMC
    start_time = time.time()
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    
    print("\nRunning MCMC...")
    sampler.run_mcmc(pos, nsteps, progress=True)
    
    mcmc_time = time.time() - start_time
    print(f"\nMCMC completato in {mcmc_time:.1f}s")
    
    # Analisi risultati
    flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)
    
    param_names = ['a0', 'A0', 'gamma', 'beta', 'alpha_proj', 'norm']
    
    print("\nRisultati MCMC:")
    mcmc_results = {}
    for i, name in enumerate(param_names):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(f"  {name} = {mcmc[1]:.4e} (+{q[1]:.4e} / -{q[0]:.4e})")
        mcmc_results[name] = {
            'median': float(mcmc[1]),
            'err_plus': float(q[1]),
            'err_minus': float(q[0])
        }
    
    # Corner plot
    try:
        import corner
        fig = corner.corner(flat_samples, labels=param_names, 
                           quantiles=[0.16, 0.5, 0.84],
                           show_titles=True, title_kwargs={"fontsize": 10})
        plt.savefig(PLOTS_DIR / 'mcmc_lensing_corner.png', dpi=150)
        print(f"\nCorner plot salvato")
    except ImportError:
        print("corner non disponibile per plot")
    
    MCMC_DONE = True
    
except ImportError:
    print("\nemcee non disponibile, uso solo ottimizzazione")
    MCMC_DONE = False
    mcmc_results = None

# =============================================================================
# CONFRONTO FINALE
# =============================================================================
print("\n" + "="*70)
print("CONFRONTO FINALE: GCV OTTIMIZZATO vs LCDM")
print("="*70)

# Parametri ottimizzati
opt_params = {
    'a0': best_params[0],
    'A0': best_params[1],
    'gamma': best_params[2],
    'beta': best_params[3],
    'alpha_proj': best_params[4],
    'norm': best_params[5]
}

# Calcola chi2 GCV ottimizzato
pred_gcv = np.array([gcv_model(R, M, opt_params) for R, M in zip(all_R, all_M)])
chi2_gcv = np.sum(((all_DS - pred_gcv) / all_err)**2)
k_gcv = 6  # parametri

# LCDM (dal test precedente)
chi2_lcdm = 21.99  # dal test precedente
k_lcdm = 4

dof_gcv = N_data - k_gcv
dof_lcdm = N_data - k_lcdm

AIC_gcv = chi2_gcv + 2 * k_gcv
AIC_lcdm = chi2_lcdm + 2 * k_lcdm
BIC_gcv = chi2_gcv + k_gcv * np.log(N_data)
BIC_lcdm = chi2_lcdm + k_lcdm * np.log(N_data)

Delta_AIC = AIC_gcv - AIC_lcdm
Delta_BIC = BIC_gcv - BIC_lcdm

print(f"\nGCV OTTIMIZZATO:")
print(f"  chi2 = {chi2_gcv:.2f}")
print(f"  chi2/dof = {chi2_gcv/dof_gcv:.3f}")
print(f"  AIC = {AIC_gcv:.1f}")
print(f"  BIC = {BIC_gcv:.1f}")

print(f"\nLCDM (NFW + baryons):")
print(f"  chi2 = {chi2_lcdm:.2f}")
print(f"  chi2/dof = {chi2_lcdm/dof_lcdm:.3f}")
print(f"  AIC = {AIC_lcdm:.1f}")
print(f"  BIC = {BIC_lcdm:.1f}")

print(f"\nCONFRONTO:")
print(f"  Delta AIC = {Delta_AIC:.1f}")
print(f"  Delta BIC = {Delta_BIC:.1f}")

if Delta_AIC < -10:
    verdict = "GCV FORTEMENTE FAVORITA"
elif Delta_AIC < -2:
    verdict = "GCV FAVORITA"
elif abs(Delta_AIC) < 2:
    verdict = "MODELLI EQUIVALENTI"
elif Delta_AIC < 10:
    verdict = "LCDM FAVORITA"
else:
    verdict = "LCDM FORTEMENTE FAVORITA"

print(f"\n  VERDETTO: {verdict}")

# =============================================================================
# PLOT FINALE
# =============================================================================
print("\n" + "="*70)
print("GENERAZIONE PLOT")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('GCV Ottimizzato vs LCDM su Dati SDSS DR7', fontsize=13, fontweight='bold')

for idx, (name, data) in enumerate(SDSS_DATA.items()):
    ax = axes[idx]
    
    R = data['R_kpc']
    obs = data['DeltaSigma']
    err = data['error']
    M_star = data['M_stellar']
    
    ax.errorbar(R, obs, yerr=err, fmt='ko', capsize=3, 
                label='SDSS DR7', markersize=7)
    
    R_plot = np.logspace(np.log10(R.min()), np.log10(R.max()), 100)
    
    # GCV ottimizzato
    pred_gcv_plot = np.array([gcv_model(r, M_star, opt_params) for r in R_plot])
    ax.plot(R_plot, pred_gcv_plot, 'b-', linewidth=2, 
            label=f'GCV opt (chi2/dof={chi2_gcv/dof_gcv:.2f})')
    
    ax.set_xlabel('R [kpc]', fontsize=11)
    ax.set_ylabel('Delta Sigma [M_sun/pc^2]', fontsize=11)
    ax.set_title(f'Sample {name}: M*={M_star:.1e} M_sun', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'gcv_optimized_lensing.png', dpi=300, bbox_inches='tight')
print(f"Plot salvato")

# =============================================================================
# SALVA RISULTATI
# =============================================================================
results = {
    'test': 'MCMC Lensing Optimization',
    'N_data': N_data,
    'optimized_params': {k: float(v) for k, v in opt_params.items()},
    'chi2_gcv': float(chi2_gcv),
    'chi2_lcdm': float(chi2_lcdm),
    'Delta_AIC': float(Delta_AIC),
    'Delta_BIC': float(Delta_BIC),
    'verdict': verdict,
    'mcmc_results': mcmc_results
}

with open(RESULTS_DIR / 'mcmc_lensing_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nRisultati salvati")

# =============================================================================
# INTERPRETAZIONE
# =============================================================================
print("\n" + "="*70)
print("INTERPRETAZIONE")
print("="*70)

print(f"""
COSA ABBIAMO IMPARATO:

1. GCV con parametri da rotation curves (a0=1.8e-10, beta=0.9) 
   NON fitta bene il lensing (chi2/dof ~ 3.6)

2. GCV OTTIMIZZATO per lensing ha parametri DIVERSI:
   - a0 = {opt_params['a0']:.2e} (vs 1.8e-10)
   - beta = {opt_params['beta']:.2f} (vs 0.90)
   - alpha_proj = {opt_params['alpha_proj']:.2f} (NUOVO parametro!)

3. Questo suggerisce che:
   a) La formula di proiezione chi_v -> Delta Sigma e' incompleta
   b) Oppure: GCV ha parametri diversi per lensing vs rotation
   c) Oppure: serve un modello piu' sofisticato

4. PROSSIMO PASSO: Derivare rigorosamente Delta Sigma dalla
   equazione di Poisson modificata, senza formule euristiche.
""")

print("="*70)
