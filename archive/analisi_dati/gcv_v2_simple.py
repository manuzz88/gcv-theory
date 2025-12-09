#!/usr/bin/env python3
"""
GCV v2.0 SEMPLIFICATO: Amplificazione Diretta del Segnale Lensing

Ipotesi: Il vuoto amplifica il ΔΣ osservato senza cambiare M_b

ΔΣ_tot(R) = ΔΣ_GCVv1(R) × [1 + η₀ × χᵥ(R)]

Questo rappresenta un contributo GEOMETRICO puro del vuoto alla curvatura
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize

# Costanti
G = 6.6743e-11
M_sun = 1.9885e30
kpc = 3.0857e19
pc = kpc / 1000

A0 = 1.72e-10
ALPHA = 2.0

print("="*70)
print("🔬 GCV v2.0: AMPLIFICAZIONE GEOMETRICA")
print("="*70)

print("""
IDEA CHIAVE:
-----------
Il vuoto NON aggiunge massa, ma amplifica la CURVATURA generata dalla massa.

Matematicamente:
  ΔΣ_osservato(R) = ΔΣ_materia(R) × [1 + η₀ × χᵥ(R)]

Dove:
- ΔΣ_materia = contributo GCV base dalla materia barionica
- η₀ × χᵥ(R) = fattore di amplificazione geometrico dal vuoto
- χᵥ(R) = suscettibilità che decresce con R

Fisicamente:
Il vuoto polarizzato crea una "lente aggiuntiva" che amplifica
la deflessione della luce oltre a quella della materia sola.

Analogo: Indice di rifrazione n > 1 in un mezzo denso
Qui: "Indice gravitazionale" n_grav = 1 + η₀ χᵥ
""")

def predict_DeltaSigma_v2(Mstar, R_kpc, eta0=0):
    """
    Predizione GCV v2 con amplificazione geometrica
    
    Parameters
    ----------
    Mstar : float [M☉]
    R_kpc : array [kpc]
    eta0 : float
        Fattore amplificazione
    """
    Mb = Mstar * M_sun
    v_inf = (G * Mb * A0)**(0.25)
    Rc = np.sqrt(G * Mb / A0) / kpc
    Rt = ALPHA * Rc
    
    DeltaSigma = np.zeros_like(R_kpc, dtype=float)
    
    for i, R in enumerate(R_kpc):
        # ΔΣ base (GCV v1)
        R_m = R * kpc
        if R < Rt:
            ds_base = v_inf**2 / (4 * G * R_m)
        else:
            ds_base = v_inf**2 / (4 * G * (Rt*kpc)) * (Rt / R)**1.7
        
        ds_base_Msun_pc2 = ds_base / (M_sun / pc**2)
        
        # Fattore amplificazione geometrico
        k = 1 / R  # kpc⁻¹
        chi_v = 1 / (1 + (k * Rc)**2)
        amplification = 1 + eta0 * chi_v
        
        # ΔΣ totale
        DeltaSigma[i] = ds_base_Msun_pc2 * amplification
    
    return DeltaSigma

# Test
Mstar = 1e11  # M☉
R_test = np.array([50, 100, 200, 400, 800])  # kpc
DeltaSigma_obs = np.array([200, 140, 80, 35, 15])  # M☉/pc²

print(f"\n📊 TEST SU M* = {Mstar:.1e} M☉")
print("="*70)

print(f"\n  R [kpc]:  {R_test}")
print(f"  Obs [M☉/pc²]: {DeltaSigma_obs}")

fig, ax = plt.subplots(figsize=(10, 7))

for eta0 in [0, 5, 10, 15, 20, 25]:
    pred = predict_DeltaSigma_v2(Mstar, R_test, eta0)
    chi2 = np.sum((DeltaSigma_obs - pred)**2 / DeltaSigma_obs)
    
    ax.plot(R_test, pred, 'o-', label=f'η₀={eta0}, χ²={chi2:.0f}', 
            linewidth=2, markersize=6)
    
    print(f"\n  η₀ = {eta0:2d}:")
    print(f"    Pred: {pred}")
    print(f"    χ² = {chi2:.1f}")

# Osservato
ax.plot(R_test, DeltaSigma_obs, 'ks-', linewidth=3, markersize=10,
        label='Osservato', zorder=10)

# Ottimizzazione
def chi2_func(eta0):
    pred = predict_DeltaSigma_v2(Mstar, R_test, eta0[0])
    return np.sum((DeltaSigma_obs - pred)**2 / DeltaSigma_obs)

result = minimize(chi2_func, [15], bounds=[(0, 50)])
eta_best = result.x[0]
chi2_best = result.fun

pred_best = predict_DeltaSigma_v2(Mstar, R_test, eta_best)

ax.plot(R_test, pred_best, 'g^-', linewidth=3, markersize=8,
        label=f'Ottimale (η₀={eta_best:.1f}), χ²={chi2_best:.1f}', zorder=9)

print(f"\n✨ OTTIMIZZAZIONE:")
print(f"  η₀ ottimale = {eta_best:.2f}")
print(f"  χ² minimo = {chi2_best:.1f}")
print(f"  Pred ottimale: {pred_best}")

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('R [kpc]', fontsize=13)
ax.set_ylabel(r'$\Delta\Sigma$ [M$_\odot$/pc$^2$]', fontsize=13)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)
ax.set_title('GCV v2: Amplificazione Geometrica del Vuoto', fontsize=12)

plt.tight_layout()
plots_dir = Path(__file__).parent / 'plots'
plots_dir.mkdir(exist_ok=True)
plt.savefig(plots_dir / 'gcv_v2_geometric_amplification.png', dpi=150)
print(f"\n💾 Plot: plots/gcv_v2_geometric_amplification.png")
plt.close()

print(f"\n{'='*70}")
print(f"📊 ANALISI RISULTATO:")
print(f"{'='*70}")

if chi2_best < 5:
    print(f"\n🎉 GCV v2 CON η₀={eta_best:.1f} MATCHA PERFETTAMENTE!")
    print(f"   χ² = {chi2_best:.1f} è eccellente")
    success = True
elif chi2_best < 20:
    print(f"\n✅ GCV v2 CON η₀={eta_best:.1f} È PLAUSIBILE")
    print(f"   χ² = {chi2_best:.1f} è accettabile")
    success = True
elif chi2_best < 100:
    print(f"\n⚠️  GCV v2 migliora ma non abbastanza")
    print(f"   χ² = {chi2_best:.1f} ancora alto")
    success = False
else:
    print(f"\n❌ GCV v2 non funziona")
    print(f"   χ² = {chi2_best:.1f} troppo alto")
    success = False

if success:
    print(f"\n💡 INTERPRETAZIONE FISICA:")
    print(f"-"*70)
    print(f"""
Il vuoto amplifica la curvatura di un fattore ~{1+eta_best:.0f}x:

A R = 50 kpc:  amplificazione = {1 + eta_best * (1/(1+(1/50*9)**2)):.1f}x
A R = 100 kpc: amplificazione = {1 + eta_best * (1/(1+(1/100*9)**2)):.1f}x
A R = 800 kpc: amplificazione = {1 + eta_best * (1/(1+(1/800*9)**2)):.1f}x

Il vuoto crea una "lente gravitazionale aggiuntiva" che:
- È forte dove χᵥ è grande (scale ~ Lc)
- Decade a grandi scale
- NON aggiunge massa (compatibile con rotazioni)
- Amplifica solo CURVATURA (spiega lensing)

PREDIZIONI TESTABILI:
1. η₀ ~ {eta_best:.0f} è parametro universale
2. Tutte le galassie mostrano stesso η₀
3. Effetto massimo a scale R ~ Lc
4. Rotazioni NON cambiano (già spiegato da a₀)
5. Cluster: lensing amplificato ma dinamica invariata

COMPATIBILITÀ:
- GCV v1: a₀ per dinamica (rotazioni, cluster timing)
- GCV v2: η₀ per geometria (lensing, curvatura)
- Stesso χᵥ(k) alla base di entrambi!
""")

print(f"{'='*70}")
