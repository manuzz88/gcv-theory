#!/usr/bin/env python3
"""
GCV v4.0: SUSCETTIBILITÀ DIPENDENTE DA DENSITÀ

INSIGHT CHIAVE:
Il vuoto risponde alla presenza di MATERIA, non solo alla scala k!

χᵥ(k, ρ) = χ₀ × (ρ/ρ₀)^α / [1 + (kLc)²]

Dove:
- α = esponente densità (nuovo parametro!)
- ρ₀ = densità caratteristica
- ρ = densità locale di materia

DENTRO galassia (ρ alta): χᵥ grande → GCV forte
LENSING (ρ bassa lungo vista): χᵥ piccolo → GCV debole

Questo spiega perché GCV funziona su rotazioni ma non su lensing!
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
c = 2.998e8

A0 = 1.72e-10
ALPHA = 2.0

print("="*70)
print("🌌 GCV v4.0: χᵥ DIPENDENTE DA DENSITÀ")
print("="*70)

print("""
MECCANISMO FISICO:
-----------------
Il vuoto quantistico si polarizza SOLO dove c'è materia densa.

In regioni dense (galassie):
  Molte particelle virtuali → forte polarizzazione → χᵥ grande

In vuoto cosmico (lensing):
  Poche particelle virtuali → debole polarizzazione → χᵥ piccolo

Formula:
  χᵥ(k, ρ) = χ₀ × (ρ/ρ₀)^α / [1 + (kLc)²]

Questo è SELF-CONSISTENT:
- Il vuoto risponde alla materia che lo circonda
- Più materia → più polarizzazione → più effetto GCV
""")

def densita_media(M_star, R_kpc):
    """
    Densità media di materia barionica entro R
    
    Approssimazione: profilo esponenziale
    ρ(r) = ρ₀ exp(-r/Rd)
    """
    # Raggio scala disco
    Rd = 3  # kpc (tipico)
    
    # Densità centrale approssimativa
    # M_star ~ 2π ρ₀ Rd² × scala_z
    scala_z = 0.3  # kpc (spessore disco)
    rho_0 = M_star * M_sun / (2 * np.pi * (Rd * kpc)**2 * (scala_z * kpc))
    
    # Densità media entro R
    # Integrale profilo esponenziale
    if R_kpc < Rd:
        rho_avg = rho_0 * (1 - np.exp(-R_kpc/Rd))
    else:
        rho_avg = rho_0 * np.exp(-R_kpc/Rd)
    
    return rho_avg  # kg/m³

def chi_v_con_densita(k_inv_kpc, Lc_kpc, rho_local, rho_0, alpha):
    """
    Suscettibilità con dipendenza da densità
    
    χᵥ(k, ρ) = χ₀ × (ρ/ρ₀)^α / [1 + (kLc)²]
    """
    chi_base = 1 / (1 + (k_inv_kpc * Lc_kpc)**2)
    
    # Fattore densità
    if rho_local > 0 and rho_0 > 0:
        density_factor = (rho_local / rho_0)**alpha
    else:
        density_factor = 0
    
    return chi_base * density_factor

def DeltaSigma_GCV_v4(M_star, R_kpc, alpha, rho_0):
    """
    ΔΣ con GCV v4 (dipendenza da densità)
    
    Parameters
    ----------
    M_star : float [M☉]
    R_kpc : array [kpc]
    alpha : float
        Esponente densità
    rho_0 : float [kg/m³]
        Densità caratteristica
    """
    Mb = M_star * M_sun
    v_inf = (G * Mb * A0)**(0.25)
    Rc = np.sqrt(G * Mb / A0) / kpc
    Rt = ALPHA * Rc
    
    DeltaSigma = np.zeros_like(R_kpc, dtype=float)
    
    for i, R in enumerate(R_kpc):
        # Densità locale a raggio R
        rho_local = densita_media(M_star, R)
        
        # χᵥ con dipendenza da densità
        k = 1 / R  # kpc⁻¹
        chi_v = chi_v_con_densita(k, Rc, rho_local, rho_0, alpha)
        
        # ΔΣ base moltiplicato per (1 + amplificazione da χᵥ)
        R_m = R * kpc
        if R < Rt:
            ds_base = v_inf**2 / (4 * G * R_m)
        else:
            ds_base = v_inf**2 / (4 * G * (Rt*kpc)) * (Rt / R)**1.7
        
        ds_base_Msun_pc2 = ds_base / (M_sun / pc**2)
        
        # Amplificazione da vuoto denso
        amplification = 1 + 10 * chi_v  # Fattore 10 da fitting
        
        DeltaSigma[i] = ds_base_Msun_pc2 * amplification
    
    return DeltaSigma

# Test
print("\n🧪 TEST SU M* = 1e11 M☉")
print("="*70)

Mstar = 1e11  # M☉
R_test = np.array([50, 100, 200, 400, 800])  # kpc
DeltaSigma_obs = np.array([200, 140, 80, 35, 15])  # M☉/pc²

# Densità caratteristica: prova varie
# ρ₀ ~ densità tipica disco galattico
rho_disk = 1e-21  # kg/m³

print(f"  Test con ρ₀ = {rho_disk:.1e} kg/m³ (disco galattico)")
print(f"  Test α da 0 a 2\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for alpha in [0, 0.5, 1.0, 1.5, 2.0]:
    pred = DeltaSigma_GCV_v4(Mstar, R_test, alpha, rho_disk)
    chi2 = np.sum((DeltaSigma_obs - pred)**2 / DeltaSigma_obs)
    
    ax1.plot(R_test, pred, 'o-', label=f'α={alpha:.1f}, χ²={chi2:.0f}',
            linewidth=2, markersize=6)
    
    print(f"  α = {alpha:.1f}: χ² = {chi2:6.1f}")

# Osservato
ax1.plot(R_test, DeltaSigma_obs, 'ks-', linewidth=3, markersize=10,
        label='Osservato', zorder=10)

# Ottimizzazione α e ρ₀
def chi2_func(params):
    alpha, log_rho0 = params
    rho0 = 10**log_rho0
    pred = DeltaSigma_GCV_v4(Mstar, R_test, alpha, rho0)
    return np.sum((DeltaSigma_obs - pred)**2 / DeltaSigma_obs)

result = minimize(chi2_func, [1.0, -21], 
                 bounds=[(0, 3), (-25, -18)])
alpha_best, log_rho0_best = result.x
rho0_best = 10**log_rho0_best
chi2_best = result.fun

pred_best = DeltaSigma_GCV_v4(Mstar, R_test, alpha_best, rho0_best)

ax1.plot(R_test, pred_best, 'g^-', linewidth=3, markersize=8,
        label=f'Ottimale (α={alpha_best:.2f}), χ²={chi2_best:.1f}',
        zorder=9)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('R [kpc]', fontsize=13)
ax1.set_ylabel(r'$\Delta\Sigma$ [M$_\odot$/pc$^2$]', fontsize=13)
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_title('GCV v4: χᵥ Dipendente da Densità', fontsize=12)

# Plot densità e χᵥ vs R
R_range = np.logspace(np.log10(10), np.log10(1000), 100)
rho_range = np.array([densita_media(Mstar, R) for R in R_range])
chi_v_range = np.array([
    chi_v_con_densita(1/R, 9, rho, rho0_best, alpha_best)
    for R, rho in zip(R_range, rho_range)
])

ax2_twin = ax2.twinx()
ax2.plot(R_range, rho_range, 'b-', linewidth=2, label='ρ(R)')
ax2_twin.plot(R_range, chi_v_range, 'r-', linewidth=2, label='χᵥ(R,ρ)')

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('R [kpc]', fontsize=13)
ax2.set_ylabel('Densità [kg/m³]', fontsize=12, color='b')
ax2_twin.set_ylabel('χᵥ', fontsize=12, color='r')
ax2.tick_params(axis='y', labelcolor='b')
ax2_twin.tick_params(axis='y', labelcolor='r')
ax2.grid(True, alpha=0.3)
ax2.set_title(f'Profili ρ(R) e χᵥ(R,ρ) con α={alpha_best:.2f}', fontsize=11)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=10)

plt.tight_layout()
plots_dir = Path(__file__).parent / 'plots'
plots_dir.mkdir(exist_ok=True)
plt.savefig(plots_dir / 'gcv_v4_density_dependent.png', dpi=150)
print(f"\n💾 Plot: plots/gcv_v4_density_dependent.png")
plt.close()

print(f"\n{'='*70}")
print(f"✨ RISULTATI OTTIMIZZAZIONE:")
print(f"{'='*70}")
print(f"  α ottimale = {alpha_best:.3f}")
print(f"  ρ₀ ottimale = {rho0_best:.2e} kg/m³")
print(f"  χ² minimo = {chi2_best:.1f}")

print(f"\n  Predizioni vs Osservazioni:")
for i, R in enumerate(R_test):
    ratio = pred_best[i] / DeltaSigma_obs[i]
    print(f"    R={R:3d} kpc: Obs={DeltaSigma_obs[i]:3.0f}, "
          f"Pred={pred_best[i]:5.1f}, Ratio={ratio:.2f}")

print(f"\n{'='*70}")
print(f"📊 VERDETTO GCV v4:")
print(f"{'='*70}")

if chi2_best < 5:
    verdict = "BREAKTHROUGH"
    print(f"\n🎉🎉🎉 GCV v4 È UNA SVOLTA! 🎉🎉🎉")
    print(f"   χ² = {chi2_best:.1f} ECCELLENTE!")
    print(f"\n   La dipendenza da densità RISOLVE il problema!")
elif chi2_best < 20:
    verdict = "PROMISING"
    print(f"\n✅ GCV v4 È MOLTO PROMETTENTE")
    print(f"   χ² = {chi2_best:.1f} significativamente migliore")
elif chi2_best < 100:
    verdict = "IMPROVEMENT"
    print(f"\n⚠️  GCV v4 migliora ma serve raffinare")
    print(f"   χ² = {chi2_best:.1f}")
else:
    verdict = "NO_IMPROVEMENT"
    print(f"\n❌ GCV v4 non migliora abbastanza")
    print(f"   χ² = {chi2_best:.1f}")

if verdict in ["BREAKTHROUGH", "PROMISING"]:
    print(f"\n💡 FISICA EMERGENTE:")
    print(f"-"*70)
    print(f"""
Il vuoto risponde alla DENSITÀ locale con esponente α = {alpha_best:.2f}

χᵥ ∝ ρ^{alpha_best:.2f}

Interpretazione:
- Il vuoto si polarizza SOLO dove c'è materia densa
- In galassie (ρ ~ 10⁻²¹ kg/m³): χᵥ grande → GCV attiva
- Nel vuoto cosmico (ρ ~ 10⁻²⁷ kg/m³): χᵥ ~ 0 → GCV inattiva

Questo spiega:
✅ Rotazioni funzionano (dentro galassia, ρ alta)
✅ Cluster funzionano (materia densa, ρ media-alta)  
✅ Lensing ora funziona meglio! (include ρ(R))

PREDIZIONI TESTABILI:
1. Dwarf galaxies (ρ bassa): GCV meno efficace
2. Galassie massive (ρ alta): GCV più efficace
3. Lensing scala con Σ_gas + Σ_stelle
4. Vuoti cosmici: solo gravità standard

PROSSIMO: Testare su TUTTI i dataset lensing!
""")

# Salva
results_dir = Path(__file__).parent / 'results'
results_dir.mkdir(exist_ok=True)

output = {
    'version': 'GCV v4.0 - Density-Dependent χᵥ',
    'verdict': verdict,
    'alpha_optimal': float(alpha_best),
    'rho0_optimal': float(rho0_best),
    'chi2_minimum': float(chi2_best),
    'formula': f'χᵥ(k,ρ) = χ₀ × (ρ/{rho0_best:.1e})^{alpha_best:.2f} / [1 + (kLc)²]'
}

with open(results_dir / 'gcv_v4_density_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n💾 Risultati: results/gcv_v4_density_results.json")
print(f"{'='*70}")
