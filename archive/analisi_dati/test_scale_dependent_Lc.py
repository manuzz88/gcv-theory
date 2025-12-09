#!/usr/bin/env python3
"""
TEST: Lc dipendente da scala (near-field vs far-field)

Ipotesi: Il vuoto ha coerenza diversa per:
- Fenomeni dinamici interni (rotazioni)
- Fenomeni geometrici proiettati (lensing)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Costanti
G = 6.6743e-11
M_sun = 1.9885e30
kpc = 3.0857e19
pc = kpc / 1000
c = 2.998e8

A0 = 1.72e-10
ALPHA = 2.0

print("="*70)
print("🧪 TEST: Lc DIPENDENTE DA SCALA")
print("="*70)

# Parametri galassia test
Mstar = 1e11 * M_sun
v_inf = (G * Mstar * A0)**(0.25)
Rc = np.sqrt(G * Mstar / A0) / kpc
Rt_base = ALPHA * Rc

print(f"\nGalassia: M* = 1e11 M☉")
print(f"  v_∞ = {v_inf/1000:.1f} km/s")
print(f"  Rc = {Rc:.1f} kpc")
print(f"  Rt_base = {Rt_base:.1f} kpc")

# Dati osservati Mandelbaum
R_obs = np.array([50, 100, 200, 400, 800])  # kpc
DeltaSigma_obs = np.array([200, 140, 80, 35, 15])  # M☉/pc²

print(f"\n📊 CONFRONTO MODELLI:")
print("="*70)

# Modello 1: Lc costante (quello attuale)
print(f"\n1️⃣  MODELLO BASE (Lc = cost = Rc)")
Rt1 = Rt_base
DeltaSigma_1 = []
for R in R_obs:
    R_m = R * kpc
    if R < Rt1:
        ds = v_inf**2 / (4 * G * R_m)
    else:
        ds = v_inf**2 / (4 * G * (Rt1*kpc)) * (Rt1 / R)**1.7
    DeltaSigma_1.append(ds / (M_sun / pc**2))
DeltaSigma_1 = np.array(DeltaSigma_1)

chi2_1 = np.sum((DeltaSigma_obs - DeltaSigma_1)**2 / DeltaSigma_obs)
print(f"  χ² = {chi2_1:.1f}")
print(f"  ΔΣ(100 kpc) = {DeltaSigma_1[1]:.1f} M☉/pc² (obs: {DeltaSigma_obs[1]})")

# Modello 2: Lc aumenta con R (far-field scaling)
print(f"\n2️⃣  MODELLO FAR-FIELD (Lc ~ √(R × Rc))")
# Lc aumenta con R osservato: Lc(R) ~ √(R × Rc)
# Questo significa Rt(R) ~ ALPHA × √(R × Rc)
DeltaSigma_2 = []
for R in R_obs:
    Lc_eff = np.sqrt(R * Rc)
    Rt_eff = ALPHA * Lc_eff
    R_m = R * kpc
    
    if R < Rt_eff:
        # Regime SIS ma con normalizzazione maggiore
        # Factor aggiuntivo da χᵥ più grande
        chi_factor = 1 + Lc_eff / Rc  # χᵥ aumenta con Lc
        ds = chi_factor * v_inf**2 / (4 * G * R_m)
    else:
        ds = v_inf**2 / (4 * G * (Rt_eff*kpc)) * (Rt_eff / R)**1.7
    DeltaSigma_2.append(ds / (M_sun / pc**2))
DeltaSigma_2 = np.array(DeltaSigma_2)

chi2_2 = np.sum((DeltaSigma_obs - DeltaSigma_2)**2 / DeltaSigma_obs)
print(f"  χ² = {chi2_2:.1f}")
print(f"  ΔΣ(100 kpc) = {DeltaSigma_2[1]:.1f} M☉/pc² (obs: {DeltaSigma_obs[1]})")

# Modello 3: Lc con boost fisso
print(f"\n3️⃣  MODELLO BOOST (Lc_lensing = 5 × Rc)")
boost = 5
Rt3 = boost * Rt_base
DeltaSigma_3 = []
for R in R_obs:
    R_m = R * kpc
    if R < Rt3:
        # Boost nella normalizzazione
        ds = boost * v_inf**2 / (4 * G * R_m)
    else:
        ds = boost * v_inf**2 / (4 * G * (Rt3*kpc)) * (Rt3 / R)**1.7
    DeltaSigma_3.append(ds / (M_sun / pc**2))
DeltaSigma_3 = np.array(DeltaSigma_3)

chi2_3 = np.sum((DeltaSigma_obs - DeltaSigma_3)**2 / DeltaSigma_obs)
print(f"  χ² = {chi2_3:.1f}")
print(f"  ΔΣ(100 kpc) = {DeltaSigma_3[1]:.1f} M☉/pc² (obs: {DeltaSigma_obs[1]})")

# Modello 4: Ottimizzazione numerica del boost
print(f"\n4️⃣  MODELLO OTTIMIZZATO (fit boost)")
from scipy.optimize import minimize

def chi2_func(params):
    boost_opt = params[0]
    Rt_opt = boost_opt * Rt_base
    pred = []
    for R in R_obs:
        R_m = R * kpc
        if R < Rt_opt:
            ds = boost_opt * v_inf**2 / (4 * G * R_m)
        else:
            ds = boost_opt * v_inf**2 / (4 * G * (Rt_opt*kpc)) * (Rt_opt / R)**1.7
        pred.append(ds / (M_sun / pc**2))
    pred = np.array(pred)
    return np.sum((DeltaSigma_obs - pred)**2 / DeltaSigma_obs)

result = minimize(chi2_func, [5.0], bounds=[(1, 20)])
boost_best = result.x[0]
chi2_best = result.fun

print(f"  Boost ottimale = {boost_best:.2f}")
print(f"  χ² = {chi2_best:.1f}")

Rt_best = boost_best * Rt_base
DeltaSigma_best = []
for R in R_obs:
    R_m = R * kpc
    if R < Rt_best:
        ds = boost_best * v_inf**2 / (4 * G * R_m)
    else:
        ds = boost_best * v_inf**2 / (4 * G * (Rt_best*kpc)) * (Rt_best / R)**1.7
    DeltaSigma_best.append(ds / (M_sun / pc**2))
DeltaSigma_best = np.array(DeltaSigma_best)

print(f"  ΔΣ(100 kpc) = {DeltaSigma_best[1]:.1f} M☉/pc² (obs: {DeltaSigma_obs[1]})")

# Plot
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(R_obs, DeltaSigma_obs, 'o-', color='black', 
        markersize=10, linewidth=2, label='Osservato (Mandelbaum)')
ax.plot(R_obs, DeltaSigma_1, 's--', color='red', 
        linewidth=2, label=f'Base (Lc=Rc), χ²={chi2_1:.0f}')
ax.plot(R_obs, DeltaSigma_2, 'd--', color='orange',
        linewidth=2, label=f'Far-field, χ²={chi2_2:.0f}')
ax.plot(R_obs, DeltaSigma_3, '^--', color='blue',
        linewidth=2, label=f'Boost=5, χ²={chi2_3:.0f}')
ax.plot(R_obs, DeltaSigma_best, 'p-', color='green',
        linewidth=2.5, label=f'Ottimizzato (boost={boost_best:.1f}), χ²={chi2_best:.0f}')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('R [kpc]', fontsize=13)
ax.set_ylabel(r'$\Delta\Sigma$ [M$_\odot$/pc$^2$]', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_title('GCV con Lc dipendente da scala - M*=1e11 M☉', fontsize=12)

plt.tight_layout()
plots_dir = Path(__file__).parent / 'plots'
plots_dir.mkdir(exist_ok=True)
plt.savefig(plots_dir / 'lensing_scale_dependent_Lc.png', dpi=150)
print(f"\n💾 Plot: plots/lensing_scale_dependent_Lc.png")
plt.close()

print(f"\n{'='*70}")
print(f"📊 RIEPILOGO:")
print(f"{'='*70}")
print(f"  Base (Lc=cost):        χ² = {chi2_1:.0f} ❌")
print(f"  Far-field (Lc~√R):     χ² = {chi2_2:.0f}")
print(f"  Boost fisso (Lc=5Rc):  χ² = {chi2_3:.0f}")
print(f"  Ottimizzato:           χ² = {chi2_best:.0f} ✅")

if chi2_best < 20:
    print(f"\n✅ CON BOOST OTTIMALE, GCV MATCHA IL LENSING!")
    print(f"   Interpretazione fisica: Lc_lensing = {boost_best:.1f} × Lc_rotations")
    print(f"   Il vuoto ha scala di coerenza maggiore per fenomeni far-field")
elif chi2_best < 50:
    print(f"\n⚠️  Migliora ma non abbastanza")
else:
    print(f"\n❌ Non basta nemmeno ottimizzando")

print(f"{'='*70}")
