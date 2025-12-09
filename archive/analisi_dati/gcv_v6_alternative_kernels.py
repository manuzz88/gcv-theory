#!/usr/bin/env python3
"""
GCV v6.0: FORME ALTERNATIVE PER χᵥ(k)

Problema identificato: χᵥ(k) = 1/(1+k²Lc²) decade troppo velocemente!

CANDIDATI PER NUOVO χᵥ(k):
========================

1. LORENTZIANO (attuale): χᵥ = 1 / [1 + (kLc)²]
   Decay: k⁻² a grandi k → TROPPO VELOCE

2. GAUSSIANO: χᵥ = exp[-(kLc)²]
   Decay: exp(-k²) → PIÙ LENTO inizialmente

3. ESPONENZIALE: χᵥ = exp[-kLc]
   Decay: exp(-k) → MOLTO PIÙ LENTO

4. POWER-LAW: χᵥ = (1 + kLc)^(-α)
   Decay: k^(-α) con α < 2 → REGOLABILE

5. LOGARITMICO: χᵥ = 1 / ln(1 + kLc/k₀)
   Decay: 1/ln(k) → LENTISSIMO

6. CUT-OFF: χᵥ = 1 se k < k_c, altrimenti decay
   Decay: Step + tail → PLATUEAU + decay

Testiamo TUTTE per trovare quale forma fisica è giusta!
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
import json

# Costanti
G = 6.6743e-11
M_sun = 1.9885e30
kpc = 3.0857e19
pc = kpc / 1000

A0 = 1.72e-10
ALPHA = 2.0

print("="*70)
print("🔬 GCV v6.0: FORME ALTERNATIVE PER χᵥ(k)")
print("="*70)

print("""
STRATEGIA:
---------
Invece di assumere forma di χᵥ(k), TESTIAMOLE TUTTE!

La fisica del vuoto quantistico potrebbe generare forme diverse
da semplice Lorentziano.

Testiamo 6 forme e vediamo quale funziona meglio.
""")

# Definizioni forme χᵥ
def chi_lorentziano(k, Lc):
    """Forma attuale: 1/(1+k²Lc²)"""
    return 1 / (1 + (k * Lc)**2)

def chi_gaussiano(k, Lc):
    """exp(-(kLc)²)"""
    return np.exp(-(k * Lc)**2)

def chi_esponenziale(k, Lc):
    """exp(-kLc)"""
    return np.exp(-k * Lc)

def chi_powerlaw(k, Lc, alpha):
    """(1 + kLc)^(-α)"""
    return (1 + k * Lc)**(-alpha)

def chi_logaritmico(k, Lc):
    """1 / ln(e + kLc)"""
    return 1 / np.log(np.e + k * Lc)

def chi_cutoff(k, Lc):
    """1 per k < 1/Lc, poi exp(-(k-kc))"""
    k_c = 1 / Lc
    result = np.ones_like(k)
    mask = k > k_c
    result[mask] = np.exp(-(k[mask] - k_c) * Lc)
    return result

def DeltaSigma_with_kernel(M_star, R_kpc, kernel_func, Lc_kpc, **kwargs):
    """
    Calcola ΔΣ con kernel χᵥ arbitrario
    """
    Mb = M_star * M_sun
    v_inf = (G * Mb * A0)**(0.25)
    Rt = ALPHA * Lc_kpc
    
    DeltaSigma = np.zeros_like(R_kpc, dtype=float)
    
    for i, R in enumerate(R_kpc):
        # ΔΣ base
        R_m = R * kpc
        if R < Rt:
            ds_base = v_inf**2 / (4 * G * R_m)
        else:
            ds_base = v_inf**2 / (4 * G * (Rt*kpc)) * (Rt / R)**1.7
        
        ds_base_Msun_pc2 = ds_base / (M_sun / pc**2)
        
        # Amplificazione da χᵥ
        k = 1 / R  # kpc⁻¹
        chi_v = kernel_func(k, Lc_kpc, **kwargs)
        
        # Amplification proporzionale a χᵥ
        amplification = 1 + 10 * chi_v  # Fattore 10 empirico
        
        DeltaSigma[i] = ds_base_Msun_pc2 * amplification
    
    return DeltaSigma

# Test
Mstar = 1e11  # M☉
Rc = np.sqrt(G * Mstar * M_sun / A0) / kpc  # ~ 9 kpc

R_test = np.array([50, 100, 200, 400, 800])  # kpc
DeltaSigma_obs = np.array([200, 140, 80, 35, 15])  # M☉/pc²

print(f"\n🧪 TEST SU M* = {Mstar:.1e} M☉")
print(f"  Rc = {Rc:.1f} kpc")
print("="*70)

# Test tutti i kernel
kernels = {
    'Lorentziano': (chi_lorentziano, {}),
    'Gaussiano': (chi_gaussiano, {}),
    'Esponenziale': (chi_esponenziale, {}),
    'Power-law α=1': (chi_powerlaw, {'alpha': 1.0}),
    'Power-law α=0.5': (chi_powerlaw, {'alpha': 0.5}),
    'Logaritmico': (chi_logaritmico, {}),
    'Cut-off': (chi_cutoff, {}),
}

results = {}

print(f"\n📊 CONFRONTO KERNEL:")
print("-"*70)

for name, (kernel_func, kwargs) in kernels.items():
    pred = DeltaSigma_with_kernel(Mstar, R_test, kernel_func, Rc, **kwargs)
    chi2 = np.sum((DeltaSigma_obs - pred)**2 / DeltaSigma_obs)
    
    results[name] = {
        'chi2': chi2,
        'pred': pred,
        'kernel': kernel_func,
        'kwargs': kwargs
    }
    
    print(f"  {name:20s}: χ² = {chi2:6.1f}")

# Trova migliore
best_kernel = min(results.items(), key=lambda x: x[1]['chi2'])
best_name = best_kernel[0]
best_chi2 = best_kernel[1]['chi2']

print(f"\n✨ MIGLIORE: {best_name} con χ² = {best_chi2:.1f}")

# Plot confronto
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Tutti i profili ΔΣ(R)
for name, data in results.items():
    ax1.plot(R_test, data['pred'], 'o-', label=f"{name} ({data['chi2']:.0f})",
            linewidth=2, markersize=5)

ax1.plot(R_test, DeltaSigma_obs, 'ks-', linewidth=3, markersize=10,
        label='Osservato', zorder=10)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('R [kpc]', fontsize=12)
ax1.set_ylabel(r'$\Delta\Sigma$ [M$_\odot$/pc$^2$]', fontsize=12)
ax1.legend(fontsize=8, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_title('Confronto Profili Lensing', fontsize=11)

# Plot 2: Forme χᵥ(k)
k_range = np.logspace(-3, 0, 100)  # kpc⁻¹

for name, data in results.items():
    chi_vals = np.array([data['kernel'](k, Rc, **data['kwargs']) for k in k_range])
    ax2.plot(k_range, chi_vals, '-', label=name, linewidth=2)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('k [kpc⁻¹]', fontsize=12)
ax2.set_ylabel('χᵥ(k)', fontsize=12)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_title('Forme χᵥ(k)', fontsize=11)
ax2.axvline(1/Rc, color='gray', linestyle='--', alpha=0.5, label=f'k=1/Rc')

# Plot 3: χᵥ(R) invece di k
R_range = np.logspace(np.log10(10), np.log10(1000), 100)  # kpc

for name, data in list(results.items())[:4]:  # Solo primi 4 per chiarezza
    chi_vals = np.array([data['kernel'](1/R, Rc, **data['kwargs']) for R in R_range])
    ax3.plot(R_range, chi_vals, '-', label=name, linewidth=2)

ax3.set_xscale('log')
ax3.set_xlabel('R [kpc]', fontsize=12)
ax3.set_ylabel('χᵥ(R)', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_title('χᵥ vs Raggio', fontsize=11)

# Plot 4: Ratio pred/obs per migliore
best_pred = results[best_name]['pred']
ratios = best_pred / DeltaSigma_obs

ax4.plot(R_test, ratios, 'go-', linewidth=2.5, markersize=10,
        label=f'{best_name}')
ax4.axhline(1, color='black', linestyle='--', linewidth=2)
ax4.fill_between(R_test, 0.8, 1.2, alpha=0.2, color='green', label='±20%')
ax4.set_xscale('log')
ax4.set_xlabel('R [kpc]', fontsize=12)
ax4.set_ylabel('Predetto / Osservato', fontsize=12)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 2)
ax4.set_title(f'Fit Quality ({best_name})', fontsize=11)

plt.tight_layout()
plots_dir = Path(__file__).parent / 'plots'
plots_dir.mkdir(exist_ok=True)
plt.savefig(plots_dir / 'gcv_v6_kernel_comparison.png', dpi=150)
print(f"\n💾 Plot: plots/gcv_v6_kernel_comparison.png")
plt.close()

# Ottimizzazione parametri per il migliore
print(f"\n{'='*70}")
print(f"🔧 OTTIMIZZAZIONE PARAMETRI PER {best_name}")
print(f"{'='*70}")

if 'alpha' in results[best_name]['kwargs']:
    # Ottimizza α per power-law
    def chi2_opt(alpha):
        pred = DeltaSigma_with_kernel(Mstar, R_test, chi_powerlaw, Rc, alpha=alpha[0])
        return np.sum((DeltaSigma_obs - pred)**2 / DeltaSigma_obs)
    
    result = minimize(chi2_opt, [1.0], bounds=[(0.1, 3.0)])
    alpha_best = result.x[0]
    chi2_best = result.fun
    
    pred_best = DeltaSigma_with_kernel(Mstar, R_test, chi_powerlaw, Rc, alpha=alpha_best)
    
    print(f"  α ottimale = {alpha_best:.3f}")
    print(f"  χ² minimo = {chi2_best:.1f}")
else:
    chi2_best = best_chi2
    pred_best = results[best_name]['pred']
    print(f"  Nessun parametro da ottimizzare")
    print(f"  χ² = {chi2_best:.1f}")

print(f"\n  Predizioni:")
for i, R in enumerate(R_test):
    ratio = pred_best[i] / DeltaSigma_obs[i]
    status = "✓" if 0.7 <= ratio <= 1.3 else "✗"
    print(f"    {status} R={R:3d} kpc: Obs={DeltaSigma_obs[i]:3.0f}, "
          f"Pred={pred_best[i]:5.1f}, Ratio={ratio:.2f}")

print(f"\n{'='*70}")
print(f"🎯 VERDETTO GCV v6:")
print(f"{'='*70}")

if chi2_best < 5:
    verdict = "BREAKTHROUGH"
    print(f"\n🎉🎉🎉 SVOLTA! KERNEL GIUSTO TROVATO! 🎉🎉🎉")
    print(f"   χ² = {chi2_best:.1f} è eccellente!")
elif chi2_best < 20:
    verdict = "SUCCESS"
    print(f"\n✅✅ GCV v6 FUNZIONA!")
    print(f"   χ² = {chi2_best:.1f} è molto buono")
elif chi2_best < 50:
    verdict = "PROMISING"
    print(f"\n✅ GCV v6 è promettente")
    print(f"   χ² = {chi2_best:.1f} migliora significativamente")
elif chi2_best < 100:
    verdict = "IMPROVEMENT"
    print(f"\n⚠️  Miglioramento ma non abbastanza")
    print(f"   χ² = {chi2_best:.1f}")
else:
    verdict = "NO_IMPROVEMENT"
    print(f"\n❌ Nessun kernel funziona")
    print(f"   χ² minimo = {chi2_best:.1f}")

if verdict in ["BREAKTHROUGH", "SUCCESS"]:
    print(f"\n💡 FISICA EMERSA:")
    print(f"-"*70)
    print(f"""
Il vuoto quantistico ha suscettibilità di forma: {best_name}!

Questo suggerisce che il vuoto NON risponde come oscillatore armonico
(che darebbe Lorentziano), ma ha dinamica diversa.

Possibili interpretazioni fisiche:
- Gaussiano: risposta con saturazione (campo non-lineare)
- Esponenziale: screening esponenziale (come Yukawa)
- Power-law: invarianza di scala (fisica critica)
- Logaritmico: flusso RG (rinormalizzazione)
- Cut-off: transizione di fase sharp

PROSSIMI PASSI CRITICI:
1. Derivare {best_name} da teoria campo vuoto
2. Testare su TUTTI i dataset lensing
3. Verificare compatibilità con rotazioni (deve dare stesso a₀!)
4. Test cosmologici (CMB, BAO)
""")

# Salva
results_dir = Path(__file__).parent / 'results'
results_dir.mkdir(exist_ok=True)

output = {
    'version': 'GCV v6.0 - Alternative Kernels',
    'verdict': verdict,
    'best_kernel': best_name,
    'chi2_minimum': float(chi2_best),
    'all_kernels': {
        name: {'chi2': float(data['chi2'])}
        for name, data in results.items()
    }
}

with open(results_dir / 'gcv_v6_kernels_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n💾 Risultati: results/gcv_v6_kernels_results.json")
print(f"{'='*70}")
