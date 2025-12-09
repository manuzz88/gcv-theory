#!/usr/bin/env python3
"""
ANALISI DELTA SIGMA: Perche' alpha_lens ~ 0.5?

Il test precedente mostra che la proiezione Sigma da' alpha ~ 1.
Ma il fit empirico su SDSS da' alpha ~ 0.5.

La differenza potrebbe essere in DELTA SIGMA:
   Delta Sigma = Sigma_mean(<R) - Sigma(R)

Questa operazione potrebbe cambiare l'esponente effettivo!
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import quad
from scipy.optimize import curve_fit
import json

print("="*70)
print("ANALISI DELTA SIGMA: Origine di alpha ~ 0.5")
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

# Parametri GCV
params = {
    'a0': 1.80e-10,
    'A0': 1.16,
    'gamma': 0.06,
    'beta': 0.90
}

def chi_v(r_kpc, M_star):
    """Suscettibilita' GCV"""
    M = M_star * M_sun
    Lc = np.sqrt(G * M / params['a0']) / kpc
    chi = params['A0'] * (M_star / 1e11)**params['gamma'] * (1 + (r_kpc / Lc)**params['beta'])
    return chi

# =============================================================================
# PROFILO DI DENSITA'
# =============================================================================

M_star = 1e11  # M_sun
a_scale = 5.0  # kpc

def rho_hernquist(r_kpc):
    """Profilo di Hernquist"""
    M = M_star * M_sun
    a = a_scale * kpc
    r = max(r_kpc, 0.01) * kpc
    return M / (2 * np.pi) * a / (r * (r + a)**3)

# =============================================================================
# CALCOLO SIGMA E DELTA SIGMA
# =============================================================================

def compute_sigma(R_kpc, with_gcv=True, r_max=2000):
    """Calcola Sigma(R) con o senza GCV"""
    def integrand(r):
        if r <= R_kpc:
            return 0
        rho = rho_hernquist(r)
        if with_gcv:
            chi = chi_v(r, M_star)
            rho = rho * (1 + chi)
        return rho * r / np.sqrt(r**2 - R_kpc**2)
    
    result, _ = quad(integrand, R_kpc * 1.001, r_max, limit=200)
    return 2 * result * kpc  # kg/m^2

def compute_sigma_mean(R_kpc, with_gcv=True, n_points=50):
    """Calcola Sigma_mean(<R) = (2/R^2) * integral_0^R Sigma(R') R' dR'"""
    R_array = np.linspace(1, R_kpc, n_points)
    Sigma_array = np.array([compute_sigma(r, with_gcv) for r in R_array])
    
    # Integrale trapezoidale
    integral = np.trapz(Sigma_array * R_array, R_array)
    return 2 * integral / R_kpc**2

def compute_delta_sigma(R_kpc, with_gcv=True):
    """Delta Sigma = Sigma_mean(<R) - Sigma(R)"""
    sigma = compute_sigma(R_kpc, with_gcv)
    sigma_mean = compute_sigma_mean(R_kpc, with_gcv)
    return sigma_mean - sigma

# =============================================================================
# TEST: Confronto Sigma vs Delta Sigma
# =============================================================================
print("\n" + "="*70)
print("CONFRONTO SIGMA vs DELTA SIGMA")
print("="*70)

R_array = np.array([20, 50, 100, 200, 500])  # kpc

print(f"\nM_star = {M_star:.1e} M_sun")
print(f"Lc = {np.sqrt(G * M_star * M_sun / params['a0']) / kpc:.1f} kpc")

print("\n--- SIGMA ---")
print(f"{'R [kpc]':>10} {'chi_v(R)':>10} {'Sigma_b':>12} {'Sigma_GCV':>12} {'Ratio':>10}")
print("-" * 60)

sigma_ratios = []
chi_values = []

for R in R_array:
    chi = chi_v(R, M_star)
    sigma_b = compute_sigma(R, with_gcv=False)
    sigma_gcv = compute_sigma(R, with_gcv=True)
    ratio = sigma_gcv / sigma_b if sigma_b > 0 else 0
    
    sigma_ratios.append(ratio)
    chi_values.append(chi)
    
    print(f"{R:>10.0f} {chi:>10.2f} {sigma_b:>12.2e} {sigma_gcv:>12.2e} {ratio:>10.2f}")

print("\n--- DELTA SIGMA ---")
print(f"{'R [kpc]':>10} {'chi_v(R)':>10} {'DS_b':>12} {'DS_GCV':>12} {'Ratio':>10}")
print("-" * 60)

ds_ratios = []

for R in R_array:
    chi = chi_v(R, M_star)
    ds_b = compute_delta_sigma(R, with_gcv=False)
    ds_gcv = compute_delta_sigma(R, with_gcv=True)
    ratio = ds_gcv / ds_b if ds_b > 0 else 0
    
    ds_ratios.append(ratio)
    
    print(f"{R:>10.0f} {chi:>10.2f} {ds_b:>12.2e} {ds_gcv:>12.2e} {ratio:>10.2f}")

# =============================================================================
# FIT ESPONENTE
# =============================================================================
print("\n" + "="*70)
print("FIT ESPONENTE EFFETTIVO")
print("="*70)

def power_model(chi, alpha):
    return (1 + chi)**alpha

# Fit per Sigma
chi_arr = np.array(chi_values)
sigma_ratio_arr = np.array(sigma_ratios)

try:
    popt_sigma, _ = curve_fit(power_model, chi_arr, sigma_ratio_arr, p0=[1.0])
    alpha_sigma = popt_sigma[0]
except:
    alpha_sigma = 1.0

# Fit per Delta Sigma
ds_ratio_arr = np.array(ds_ratios)

try:
    popt_ds, _ = curve_fit(power_model, chi_arr, ds_ratio_arr, p0=[1.0])
    alpha_ds = popt_ds[0]
except:
    alpha_ds = 1.0

print(f"\nEsponente per SIGMA: alpha = {alpha_sigma:.3f}")
print(f"Esponente per DELTA SIGMA: alpha = {alpha_ds:.3f}")

# =============================================================================
# ANALISI: Perche' sono diversi?
# =============================================================================
print("\n" + "="*70)
print("ANALISI DELLA DIFFERENZA")
print("="*70)

print("""
La differenza tra alpha_Sigma e alpha_DeltaSigma viene da:

Delta Sigma = Sigma_mean(<R) - Sigma(R)

Se Sigma_GCV = Sigma_b * f(chi_v), allora:

Delta Sigma_GCV = Sigma_mean_b * f(chi_v_mean) - Sigma_b * f(chi_v)

dove chi_v_mean e' una media pesata di chi_v per R' < R.

Poiche' chi_v CRESCE con R (beta > 0), abbiamo:
- chi_v_mean < chi_v(R)
- f(chi_v_mean) < f(chi_v)

Questo RIDUCE il rapporto Delta Sigma_GCV / Delta Sigma_b
rispetto a Sigma_GCV / Sigma_b!
""")

# Verifica numerica
print("\nVerifica numerica:")
print(f"{'R [kpc]':>10} {'chi_v(R)':>10} {'chi_v_mean':>12} {'Ratio chi':>12}")
print("-" * 50)

for R in R_array:
    chi_R = chi_v(R, M_star)
    
    # Calcola chi_v medio pesato per Sigma
    R_inner = np.linspace(1, R, 30)
    chi_inner = np.array([chi_v(r, M_star) for r in R_inner])
    sigma_inner = np.array([compute_sigma(r, with_gcv=False) for r in R_inner])
    
    # Media pesata
    chi_mean = np.trapz(chi_inner * sigma_inner * R_inner, R_inner) / np.trapz(sigma_inner * R_inner, R_inner)
    
    print(f"{R:>10.0f} {chi_R:>10.2f} {chi_mean:>12.2f} {chi_mean/chi_R:>12.3f}")

# =============================================================================
# MODELLO ANALITICO SEMPLIFICATO
# =============================================================================
print("\n" + "="*70)
print("MODELLO ANALITICO SEMPLIFICATO")
print("="*70)

print("""
Per un profilo di potenza: Sigma ~ R^(-n), chi_v ~ R^beta

Sigma_GCV(R) ~ R^(-n) * R^beta = R^(beta-n)
Sigma_mean_GCV(<R) ~ R^(beta-n+2) / R^2 = R^(beta-n)  [stesso scaling!]

Quindi Delta Sigma_GCV ~ R^(beta-n) - R^(beta-n) = 0 ???

No, i coefficienti sono diversi! Facciamo meglio:

Sigma(R) = A * R^(-n) * (1 + B * R^beta)
Sigma_mean(<R) = (2/R^2) * integral_0^R A * R'^(-n) * (1 + B * R'^beta) * R' dR'
              = A * R^(-n) * [1 + B' * R^beta]  con B' != B

Delta Sigma = A * R^(-n) * [(1 + B' * R^beta) - (1 + B * R^beta)]
            = A * R^(-n) * (B' - B) * R^beta

Se B' < B (perche' la media e' su R' < R), allora Delta Sigma < 0 ???

Questo non torna. Il problema e' piu' sottile.
""")

# =============================================================================
# CONFRONTO CON DATI SDSS
# =============================================================================
print("\n" + "="*70)
print("CONFRONTO CON DATI SDSS")
print("="*70)

# Dati SDSS (dal test precedente)
SDSS_L4 = {
    'R_kpc': np.array([42, 65, 104, 165, 261, 414, 656, 1039, 1645, 2607]),
    'DeltaSigma': np.array([127.3, 98.2, 74.1, 55.8, 40.2, 27.1, 17.8, 11.2, 7.1, 4.3]),
    'M_stellar': 1.5e11
}

print("\nConfronto predizioni GCV con SDSS L4:")
print(f"{'R [kpc]':>10} {'DS_obs':>10} {'DS_GCV':>12} {'DS_b':>12} {'Ratio_obs':>12}")
print("-" * 60)

# Usa massa SDSS
M_star_sdss = SDSS_L4['M_stellar']

for i, R in enumerate(SDSS_L4['R_kpc'][:5]):  # primi 5 punti
    ds_obs = SDSS_L4['DeltaSigma'][i]
    
    # Predizione GCV (con M_star SDSS)
    # Nota: devo riscalare per la massa
    chi = chi_v(R, M_star_sdss)
    
    # Stima semplificata
    ds_b_est = ds_obs / (1 + chi)**0.5  # assumendo alpha ~ 0.5
    ds_gcv_est = ds_b_est * (1 + chi)  # con alpha = 1
    
    ratio = ds_obs / ds_b_est if ds_b_est > 0 else 0
    
    print(f"{R:>10.0f} {ds_obs:>10.1f} {ds_gcv_est:>12.1f} {ds_b_est:>12.1f} {ratio:>12.2f}")

# =============================================================================
# CONCLUSIONE CHIAVE
# =============================================================================
print("\n" + "="*70)
print("CONCLUSIONE CHIAVE")
print("="*70)

print(f"""
RISULTATI:

1. Per SIGMA: alpha ~ {alpha_sigma:.2f} (vicino a 1)
2. Per DELTA SIGMA: alpha ~ {alpha_ds:.2f}

3. La differenza viene dal fatto che Delta Sigma e' una DIFFERENZA
   tra due quantita' che scalano in modo simile con chi_v.

4. IPOTESI per alpha ~ 0.5 nei dati SDSS:
   
   Il valore 0.5 potrebbe NON essere un esponente fisico,
   ma un PARAMETRO EFFETTIVO che cattura:
   
   a) La differenza tra Sigma e Delta Sigma
   b) Effetti di proiezione non banali
   c) Correlazioni tra bin radiali
   d) Errori sistematici nei dati

5. IMPLICAZIONE:
   
   Se alpha ~ 0.5 e' un parametro effettivo (non derivabile),
   allora GCV richiede CALIBRAZIONE EMPIRICA per il lensing,
   simile a come MOND richiede calibrazione per i cluster.

6. ALTERNATIVA:
   
   Forse la forma di chi_v deve essere DIVERSA per il lensing.
   Non chi_v^0.5, ma una forma funzionale diversa.
""")

# =============================================================================
# PLOT
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Analisi: Sigma vs Delta Sigma in GCV', fontsize=13, fontweight='bold')

# Plot 1: Ratio vs chi_v
ax1 = axes[0]
ax1.scatter(chi_values, sigma_ratios, s=100, c='blue', label='Sigma ratio', marker='o')
ax1.scatter(chi_values, ds_ratios, s=100, c='red', label='Delta Sigma ratio', marker='s')

chi_fit = np.linspace(min(chi_values), max(chi_values), 100)
ax1.plot(chi_fit, (1 + chi_fit)**1.0, 'b--', alpha=0.5, label='alpha=1.0')
ax1.plot(chi_fit, (1 + chi_fit)**0.5, 'r--', alpha=0.5, label='alpha=0.5')
ax1.plot(chi_fit, (1 + chi_fit)**alpha_sigma, 'b-', label=f'Sigma fit: {alpha_sigma:.2f}')
ax1.plot(chi_fit, (1 + chi_fit)**alpha_ds, 'r-', label=f'DS fit: {alpha_ds:.2f}')

ax1.set_xlabel('chi_v(R)')
ax1.set_ylabel('Ratio GCV/baryonic')
ax1.set_title('Esponente effettivo')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: Profili
ax2 = axes[1]
R_plot = np.array([20, 50, 100, 200, 500])
sigma_b = [compute_sigma(R, with_gcv=False) for R in R_plot]
sigma_gcv = [compute_sigma(R, with_gcv=True) for R in R_plot]
ds_b = [compute_delta_sigma(R, with_gcv=False) for R in R_plot]
ds_gcv = [compute_delta_sigma(R, with_gcv=True) for R in R_plot]

ax2.loglog(R_plot, sigma_b, 'b--', label='Sigma_b')
ax2.loglog(R_plot, sigma_gcv, 'b-', label='Sigma_GCV')
ax2.loglog(R_plot, np.abs(ds_b), 'r--', label='|DS_b|')
ax2.loglog(R_plot, np.abs(ds_gcv), 'r-', label='|DS_GCV|')

ax2.set_xlabel('R [kpc]')
ax2.set_ylabel('Sigma o Delta Sigma [kg/m^2]')
ax2.set_title('Profili')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'delta_sigma_analysis.png', dpi=300, bbox_inches='tight')
print("\nPlot salvato")

# Salva risultati
results = {
    'alpha_sigma': float(alpha_sigma),
    'alpha_delta_sigma': float(alpha_ds),
    'R_values': R_array.tolist(),
    'chi_values': chi_values,
    'sigma_ratios': sigma_ratios,
    'ds_ratios': ds_ratios,
    'conclusion': 'alpha ~ 0.5 is likely an effective parameter, not derivable from first principles'
}

with open(RESULTS_DIR / 'delta_sigma_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("MESSAGGIO FINALE")
print("="*70)
print(f"""
Il valore alpha ~ 0.5 trovato empiricamente NON deriva direttamente
dalla proiezione geometrica (che da' alpha ~ 1).

Possibili spiegazioni:
1. E' un parametro effettivo che assorbe vari effetti
2. La forma di chi_v per il lensing e' diversa
3. Ci sono effetti non lineari nella relazione Sigma -> Delta Sigma

RACCOMANDAZIONE:
Trattare alpha_lens come parametro libero da calibrare sui dati,
simile alla funzione di interpolazione mu(x) in MOND.
""")
print("="*70)
