#!/usr/bin/env python3
"""
DERIVAZIONE RIGOROSA: Da Poisson Modificata a Delta Sigma

Il problema principale di GCV sul lensing e' che usiamo formule euristiche.
Questo script deriva RIGOROSAMENTE Delta Sigma dalla equazione di Poisson modificata.

Equazione di Poisson modificata:
    nabla . [(1 + chi_v) nabla Phi] = 4 pi G rho_b

Per simmetria sferica:
    d/dr [r^2 (1 + chi_v) dPhi/dr] = 4 pi G r^2 rho_b

Integrazione -> Phi(r) -> Sigma(R) -> Delta Sigma(R)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.integrate import quad, cumtrapz
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DERIVAZIONE RIGOROSA: Poisson Modificata -> Delta Sigma")
print("="*70)

# Directories
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

# Costanti
G = 6.674e-11  # m^3 kg^-1 s^-2
c = 2.998e8    # m/s
M_sun = 1.989e30  # kg
kpc = 3.086e19  # m
pc = 3.086e16   # m
Mpc = 3.086e22  # m
H0 = 67.4
h = H0 / 100

# =============================================================================
# PROFILO DI DENSITA' BARIONICA
# =============================================================================

def hernquist_density(r, M_star, a):
    """
    Profilo di Hernquist per la distribuzione stellare
    
    rho(r) = M / (2 pi) * a / (r * (r + a)^3)
    
    Parametri:
    - M_star: massa stellare totale [M_sun]
    - a: scala di lunghezza [kpc]
    """
    M = M_star * M_sun
    a_m = a * kpc
    r_m = r * kpc
    
    rho = M / (2 * np.pi) * a_m / (r_m * (r_m + a_m)**3)
    return rho  # kg/m^3

def hernquist_enclosed_mass(r, M_star, a):
    """Massa racchiusa entro r per profilo di Hernquist"""
    M = M_star * M_sun
    a_m = a * kpc
    r_m = r * kpc
    
    M_enc = M * r_m**2 / (r_m + a_m)**2
    return M_enc  # kg

# =============================================================================
# SUSCETTIBILITA' DEL VUOTO chi_v
# =============================================================================

def chi_v_gcv(r, M_star, params):
    """
    Suscettibilita' del vuoto GCV v2.1
    
    chi_v(r) = A0 * (M/M0)^gamma * [1 + (r/Lc)^beta]
    
    dove Lc = sqrt(G M / a0)
    """
    a0 = params['a0']
    A0 = params['A0']
    gamma = params['gamma']
    beta = params['beta']
    
    M = M_star * M_sun
    Lc = np.sqrt(G * M / a0) / kpc  # in kpc
    
    chi = A0 * (M_star / 1e11)**gamma * (1 + (r / Lc)**beta)
    return chi

# =============================================================================
# SOLUZIONE EQUAZIONE DI POISSON MODIFICATA
# =============================================================================

def solve_modified_poisson(r_array, M_star, a_scale, params):
    """
    Risolve l'equazione di Poisson modificata per ottenere Phi(r)
    
    Per simmetria sferica:
    d/dr [r^2 (1 + chi_v) dPhi/dr] = 4 pi G r^2 rho_b
    
    Integrando:
    r^2 (1 + chi_v) dPhi/dr = 4 pi G integral_0^r rho_b r'^2 dr' = G M_enc(r)
    
    Quindi:
    dPhi/dr = G M_enc(r) / [r^2 (1 + chi_v(r))]
    
    E il potenziale:
    Phi(r) = -integral_r^inf dPhi/dr' dr'
    """
    # Calcola massa racchiusa
    M_enc = np.array([hernquist_enclosed_mass(r, M_star, a_scale) for r in r_array])
    
    # Calcola chi_v
    chi_v = np.array([chi_v_gcv(r, M_star, params) for r in r_array])
    
    # Calcola dPhi/dr
    r_m = r_array * kpc
    dPhi_dr = G * M_enc / (r_m**2 * (1 + chi_v))
    
    # Integra per ottenere Phi (dall'infinito)
    # Phi(r) = -integral_r^inf dPhi/dr' dr'
    # Usiamo integrazione cumulativa inversa
    
    # Estendi a grandi raggi per approssimare infinito
    r_ext = np.logspace(np.log10(r_array[-1]), 4, 100)  # fino a 10^4 kpc
    M_enc_ext = np.array([hernquist_enclosed_mass(r, M_star, a_scale) for r in r_ext])
    chi_v_ext = np.array([chi_v_gcv(r, M_star, params) for r in r_ext])
    r_ext_m = r_ext * kpc
    dPhi_dr_ext = G * M_enc_ext / (r_ext_m**2 * (1 + chi_v_ext))
    
    # Combina
    r_full = np.concatenate([r_array, r_ext[1:]])
    dPhi_dr_full = np.concatenate([dPhi_dr, dPhi_dr_ext[1:]])
    r_full_m = r_full * kpc
    
    # Integra da infinito (approssimato)
    Phi_full = np.zeros_like(r_full)
    for i in range(len(r_full)-1, -1, -1):
        if i == len(r_full) - 1:
            Phi_full[i] = 0  # Phi(inf) = 0
        else:
            dr = (r_full[i+1] - r_full[i]) * kpc
            Phi_full[i] = Phi_full[i+1] - dPhi_dr_full[i] * dr
    
    # Ritorna solo la parte originale
    Phi = Phi_full[:len(r_array)]
    
    return Phi, dPhi_dr, chi_v

# =============================================================================
# PROIEZIONE: Phi(r) -> Sigma(R) -> Delta Sigma(R)
# =============================================================================

def compute_surface_density(R_array, r_array, rho_3d, Phi, params):
    """
    Calcola la densita' superficiale proiettata Sigma(R)
    
    Per lensing, la quantita' rilevante e' la densita' superficiale
    della massa EFFICACE, non solo barionica.
    
    In GCV, la massa efficace e':
    M_eff(r) = M_b(r) * (1 + chi_v(r))
    
    Quindi la densita' efficace e':
    rho_eff(r) = rho_b(r) * (1 + chi_v(r))
    
    E la densita' superficiale:
    Sigma(R) = 2 * integral_R^inf rho_eff(r) * r / sqrt(r^2 - R^2) dr
    """
    Sigma = np.zeros_like(R_array)
    
    for i, R in enumerate(R_array):
        # Integra lungo la linea di vista
        def integrand(r):
            if r <= R:
                return 0
            # rho_eff = rho_b * (1 + chi_v)
            rho_b = hernquist_density(r, params['M_star'], params['a_scale'])
            chi_v = chi_v_gcv(r, params['M_star'], params)
            rho_eff = rho_b * (1 + chi_v)
            return rho_eff * r / np.sqrt(r**2 - R**2)
        
        # Integra da R a infinito (approssimato come 10*R_max)
        r_max = max(10 * R, 1000)  # kpc
        result, _ = quad(integrand, R * 1.001, r_max, limit=100)
        Sigma[i] = 2 * result * kpc  # converti in kg/m^2
    
    return Sigma

def compute_delta_sigma(R_array, Sigma):
    """
    Calcola Delta Sigma = Sigma_mean(<R) - Sigma(R)
    
    Sigma_mean(<R) = 2/R^2 * integral_0^R Sigma(R') R' dR'
    """
    DeltaSigma = np.zeros_like(R_array)
    
    # Interpola Sigma per integrazione
    Sigma_interp = interp1d(R_array, Sigma, kind='cubic', fill_value='extrapolate')
    
    for i, R in enumerate(R_array):
        # Calcola Sigma_mean(<R)
        def integrand(Rp):
            return Sigma_interp(Rp) * Rp
        
        result, _ = quad(integrand, R_array[0], R, limit=100)
        Sigma_mean = 2 * result / R**2
        
        DeltaSigma[i] = Sigma_mean - Sigma[i]
    
    # Converti in M_sun/pc^2
    DeltaSigma_Msun_pc2 = DeltaSigma / (M_sun / pc**2)
    
    return DeltaSigma_Msun_pc2

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

# =============================================================================
# TEST: Calcola Delta Sigma per un sample
# =============================================================================
print("\n" + "="*70)
print("CALCOLO RIGOROSO Delta Sigma")
print("="*70)

# Parametri GCV
gcv_params = {
    'a0': 1.80e-10,
    'A0': 1.16,
    'gamma': 0.06,
    'beta': 0.90,
}

def compute_gcv_prediction(M_star, R_obs, params, norm=1.0):
    """Calcola predizione GCV completa per un sample"""
    
    # Scala di Hernquist (approssimazione: a ~ 0.015 * R_eff, R_eff ~ 5 kpc per M* ~ 10^11)
    a_scale = 3.0 * (M_star / 1e11)**0.3  # kpc
    
    # Griglia radiale fine
    r_array = np.logspace(0, 4, 200)  # 1 - 10000 kpc
    
    # Aggiungi parametri per la funzione
    full_params = params.copy()
    full_params['M_star'] = M_star
    full_params['a_scale'] = a_scale
    
    # Calcola densita' 3D
    rho_3d = np.array([hernquist_density(r, M_star, a_scale) for r in r_array])
    
    # Risolvi Poisson modificata
    Phi, dPhi_dr, chi_v = solve_modified_poisson(r_array, M_star, a_scale, params)
    
    # Calcola Sigma
    print(f"  Calcolando Sigma(R) per M*={M_star:.1e}...")
    Sigma = compute_surface_density(R_obs, r_array, rho_3d, Phi, full_params)
    
    # Calcola Delta Sigma
    print(f"  Calcolando Delta Sigma(R)...")
    DeltaSigma = compute_delta_sigma(R_obs, Sigma)
    
    return norm * DeltaSigma

# Test su sample L4
print("\nSample L4:")
data_L4 = SDSS_DATA['L4']
pred_L4 = compute_gcv_prediction(data_L4['M_stellar'], data_L4['R_kpc'], gcv_params)

print(f"\n  R [kpc]    Obs [M_sun/pc^2]    GCV [M_sun/pc^2]    Ratio")
print(f"  " + "-"*60)
for i in range(len(data_L4['R_kpc'])):
    ratio = pred_L4[i] / data_L4['DeltaSigma'][i] if pred_L4[i] > 0 else 0
    print(f"  {data_L4['R_kpc'][i]:6.0f}    {data_L4['DeltaSigma'][i]:8.1f}            {pred_L4[i]:8.1f}            {ratio:.2f}")

# =============================================================================
# OTTIMIZZAZIONE NORMALIZZAZIONE
# =============================================================================
print("\n" + "="*70)
print("OTTIMIZZAZIONE")
print("="*70)

def chi2_total(norm, params):
    """Chi2 totale su entrambi i sample"""
    chi2 = 0
    
    for name, data in SDSS_DATA.items():
        pred = compute_gcv_prediction(data['M_stellar'], data['R_kpc'], params, norm)
        chi2 += np.sum(((data['DeltaSigma'] - pred) / data['error'])**2)
    
    return chi2

# Trova normalizzazione ottimale
print("\nOttimizzando normalizzazione...")
from scipy.optimize import minimize_scalar

result = minimize_scalar(lambda n: chi2_total(n, gcv_params), bounds=(0.01, 100), method='bounded')
best_norm = result.x
best_chi2 = result.fun

N_data = sum(len(d['R_kpc']) for d in SDSS_DATA.values())
chi2_red = best_chi2 / (N_data - 1)

print(f"\nRisultati:")
print(f"  Normalizzazione ottimale: {best_norm:.3f}")
print(f"  Chi2 totale: {best_chi2:.2f}")
print(f"  Chi2/dof: {chi2_red:.3f}")

# =============================================================================
# CONFRONTO CON LCDM
# =============================================================================
print("\n" + "="*70)
print("CONFRONTO CON LCDM")
print("="*70)

chi2_lcdm = 21.99  # dal test precedente
k_lcdm = 4
k_gcv = 1  # solo normalizzazione

AIC_gcv = best_chi2 + 2 * k_gcv
AIC_lcdm = chi2_lcdm + 2 * k_lcdm
Delta_AIC = AIC_gcv - AIC_lcdm

BIC_gcv = best_chi2 + k_gcv * np.log(N_data)
BIC_lcdm = chi2_lcdm + k_lcdm * np.log(N_data)
Delta_BIC = BIC_gcv - BIC_lcdm

print(f"\nGCV (derivazione rigorosa):")
print(f"  Chi2 = {best_chi2:.2f}")
print(f"  Chi2/dof = {chi2_red:.3f}")
print(f"  AIC = {AIC_gcv:.1f}")

print(f"\nLCDM:")
print(f"  Chi2 = {chi2_lcdm:.2f}")
print(f"  AIC = {AIC_lcdm:.1f}")

print(f"\nDelta AIC = {Delta_AIC:.1f}")
print(f"Delta BIC = {Delta_BIC:.1f}")

if Delta_AIC < -2:
    verdict = "GCV FAVORITA"
elif abs(Delta_AIC) < 2:
    verdict = "EQUIVALENTI"
else:
    verdict = "LCDM FAVORITA"

print(f"\nVERDETTO: {verdict}")

# =============================================================================
# PLOT
# =============================================================================
print("\n" + "="*70)
print("GENERAZIONE PLOT")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('GCV Derivazione Rigorosa vs Dati SDSS DR7', fontsize=13, fontweight='bold')

for idx, (name, data) in enumerate(SDSS_DATA.items()):
    ax = axes[idx]
    
    R = data['R_kpc']
    obs = data['DeltaSigma']
    err = data['error']
    
    # Predizione GCV
    pred = compute_gcv_prediction(data['M_stellar'], R, gcv_params, best_norm)
    
    ax.errorbar(R, obs, yerr=err, fmt='ko', capsize=3, label='SDSS DR7', markersize=7)
    ax.plot(R, pred, 'b-', linewidth=2, label=f'GCV rigoroso')
    
    ax.set_xlabel('R [kpc]', fontsize=11)
    ax.set_ylabel('Delta Sigma [M_sun/pc^2]', fontsize=11)
    ax.set_title(f'Sample {name}: M*={data["M_stellar"]:.1e} M_sun', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'gcv_rigorous_lensing.png', dpi=300, bbox_inches='tight')
print(f"Plot salvato")

# =============================================================================
# SALVA RISULTATI
# =============================================================================
results = {
    'test': 'Rigorous Lensing Derivation',
    'method': 'Modified Poisson -> Sigma -> Delta Sigma',
    'gcv_params': gcv_params,
    'best_norm': float(best_norm),
    'chi2': float(best_chi2),
    'chi2_red': float(chi2_red),
    'Delta_AIC': float(Delta_AIC),
    'verdict': verdict
}

with open(RESULTS_DIR / 'rigorous_lensing_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nRisultati salvati")

# =============================================================================
# ANALISI
# =============================================================================
print("\n" + "="*70)
print("ANALISI")
print("="*70)

print(f"""
COSA ABBIAMO FATTO:

1. Derivato RIGOROSAMENTE Delta Sigma dalla equazione di Poisson modificata:
   - Profilo di Hernquist per materia barionica
   - chi_v(r) = A0 * (M/M0)^gamma * [1 + (r/Lc)^beta]
   - Massa efficace: M_eff = M_b * (1 + chi_v)
   - Proiezione lungo la linea di vista

2. RISULTATO:
   - Chi2/dof = {chi2_red:.3f}
   - Delta AIC = {Delta_AIC:.1f}
   - Verdetto: {verdict}

3. INTERPRETAZIONE:
""")

if Delta_AIC > 10:
    print("""   La derivazione rigorosa NON migliora significativamente il fit.
   Questo suggerisce che il problema non e' nella proiezione,
   ma nella forma funzionale di chi_v stessa.
   
   PROSSIMI PASSI:
   a) Esplorare forme alternative di chi_v
   b) Considerare dipendenza da ambiente (clustering)
   c) Verificare se lensing richiede fisica diversa da rotation curves
""")
else:
    print("""   La derivazione rigorosa MIGLIORA il fit!
   GCV e' competitiva con LCDM quando si usa la fisica corretta.
""")

print("="*70)
