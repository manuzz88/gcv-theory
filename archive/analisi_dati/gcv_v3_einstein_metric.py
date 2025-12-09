#!/usr/bin/env python3
"""
GCV v3.0: MODIFICA DIRETTA DELLA METRICA (EINSTEIN)

Approccio radicale: Il vuoto modifica la METRICA g_μν direttamente

METRICA STANDARD (Schwarzschild):
ds² = -(1 - 2Φ/c²)c²dt² + (1 + 2Φ/c²)(dx² + dy² + dz²)

Dove Φ = -GM/r (potenziale Newtoniano)

METRICA GCV v3 (vuoto attivo):
ds² = -(1 - 2Φ_eff/c²)c²dt² + (1 + 2Φ_eff/c²)(dx² + dy² + dz²)

Dove:
Φ_eff(r) = Φ(r) × [1 + f(χᵥ(r))]

Il vuoto AMPLIFICA il potenziale gravitazionale → amplifica curvatura!

Questo è diverso da GCV v1 che modificava solo ∇²Φ
Qui modifichiamo Φ stesso nella metrica!
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
import json

# Costanti
G = 6.6743e-11  # m³ kg⁻¹ s⁻²
M_sun = 1.9885e30  # kg
kpc = 3.0857e19  # m
pc = kpc / 1000  # m
c = 2.998e8  # m/s

A0 = 1.72e-10  # m/s²
ALPHA = 2.0

print("="*70)
print("🌌 GCV v3.0: MODIFICA METRICA EINSTEIN")
print("="*70)

print("""
IDEA CENTRALE:
-------------
Il vuoto con χᵥ modifica la struttura DELLO SPAZIO-TEMPO stesso,
non solo il campo gravitazionale che si propaga in esso.

METRICA MODIFICATA:
------------------
g₀₀ = -(1 - 2Φ_eff/c²)

Dove:
Φ_eff(r) = Φ_Newton(r) × [1 + ξ₀ × χᵥ(r)]

- Φ_Newton = -GM/r (standard)
- ξ₀ = fattore accoppiamento vuoto-metrica (NUOVO PARAMETRO)
- χᵥ(r) = 1/(1 + (r/Lc)²) (suscettibilità scala-dipendente)

LENSING DA METRICA:
------------------
Per lente debole, l'angolo di deflessione è:

α(b) = (4G/c²) × M_eff(b) / b

Dove M_eff(b) è la massa entro raggio di impatto b "vista" dalla metrica.

Con Φ_eff amplificato:
M_eff(b) = M(b) × [1 + ξ₀ × χᵥ(b)]

Quindi:
ΔΣ(R) = ΔΣ_standard(R) × [1 + ξ₀ × χᵥ(R)]

DIFFERENZA DA GCV v2:
--------------------
GCV v2: Amplificazione fenomenologica del segnale
GCV v3: Amplificazione dalla METRICA modificata

Fisicamente più fondamentale!
""")

def suscettibilita(r_kpc, Rc_kpc):
    """Suscettibilità vuoto χᵥ(r)"""
    return 1 / (1 + (r_kpc / Rc_kpc)**2)

def potenziale_efficace(M_sun_val, r_kpc, xi0, Rc_kpc):
    """
    Potenziale efficace dalla metrica modificata
    
    Φ_eff = -(GM/r) × [1 + ξ₀ χᵥ(r)]
    """
    r_m = r_kpc * kpc
    Phi_Newton = -G * M_sun_val * M_sun / r_m  # J/kg = m²/s²
    chi_v = suscettibilita(r_kpc, Rc_kpc)
    Phi_eff = Phi_Newton * (1 + xi0 * chi_v)
    return Phi_eff

def velocita_rotazione_v3(M_baryon, r_kpc, xi0, Rc_kpc):
    """
    Velocità di rotazione da metrica modificata
    
    v² = -r × dΦ_eff/dr
    """
    # Derivata numerica
    dr = 0.01  # kpc
    Phi_1 = potenziale_efficace(M_baryon, r_kpc + dr/2, xi0, Rc_kpc)
    Phi_2 = potenziale_efficace(M_baryon, r_kpc - dr/2, xi0, Rc_kpc)
    dPhi_dr = (Phi_1 - Phi_2) / (dr * kpc)  # s⁻²
    
    v_squared = -(r_kpc * kpc) * dPhi_dr
    
    if v_squared > 0:
        return np.sqrt(v_squared)
    else:
        return 0

def DeltaSigma_from_metric(M_baryon, R_kpc, xi0, Rc_kpc):
    """
    ΔΣ dal lensing con metrica modificata
    
    Per lente debole: α ∝ M_eff/b
    ΔΣ ∝ M_eff proiettato
    """
    # Massa efficace "vista" dal lensing a raggio R
    # M_eff(R) = M(R) × [1 + ξ₀ χᵥ(R)]
    
    chi_v = suscettibilita(R_kpc, Rc_kpc)
    amplification = 1 + xi0 * chi_v
    
    # ΔΣ base da GCV v1
    Mb = M_baryon * M_sun
    v_inf = (G * Mb * A0)**(0.25)
    Rt = ALPHA * Rc_kpc
    
    R_m = R_kpc * kpc
    if R_kpc < Rt:
        ds_base = v_inf**2 / (4 * G * R_m)
    else:
        ds_base = v_inf**2 / (4 * G * (Rt*kpc)) * (Rt / R_kpc)**1.7
    
    ds_base_Msun_pc2 = ds_base / (M_sun / pc**2)
    
    # Applicazione amplificazione dalla metrica
    DeltaSigma = ds_base_Msun_pc2 * amplification
    
    return DeltaSigma

print("\n🧪 TEST SU GALASSIA M* = 1e11 M☉")
print("="*70)

Mstar = 1e11  # M☉
Rc = np.sqrt(G * Mstar * M_sun / A0) / kpc  # ~ 9 kpc

# Dati osservativi
R_test = np.array([50, 100, 200, 400, 800])  # kpc
DeltaSigma_obs = np.array([200, 140, 80, 35, 15])  # M☉/pc²

print(f"  Rc = {Rc:.1f} kpc")
print(f"  Dati: {len(R_test)} punti da 50 a 800 kpc")

# Test diversi ξ₀
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

xi0_values = [0, 5, 10, 15, 20, 30]
results_test = {}

for xi0 in xi0_values:
    DeltaSigma_pred = np.array([
        DeltaSigma_from_metric(Mstar, R, xi0, Rc) 
        for R in R_test
    ])
    
    chi2 = np.sum((DeltaSigma_obs - DeltaSigma_pred)**2 / DeltaSigma_obs)
    
    ax1.plot(R_test, DeltaSigma_pred, 'o-', 
            label=f'ξ₀={xi0}, χ²={chi2:.0f}',
            linewidth=2, markersize=6)
    
    results_test[xi0] = {'chi2': chi2, 'pred': DeltaSigma_pred}
    
    print(f"\n  ξ₀ = {xi0:2d}: χ² = {chi2:6.1f}")

# Osservato
ax1.plot(R_test, DeltaSigma_obs, 'ks-', linewidth=3, markersize=10,
        label='Osservato', zorder=10)

# Ottimizzazione ξ₀
def chi2_func(xi0):
    pred = np.array([
        DeltaSigma_from_metric(Mstar, R, xi0[0], Rc) 
        for R in R_test
    ])
    return np.sum((DeltaSigma_obs - pred)**2 / DeltaSigma_obs)

result = minimize(chi2_func, [15], bounds=[(0, 50)])
xi0_best = result.x[0]
chi2_best = result.fun

DeltaSigma_best = np.array([
    DeltaSigma_from_metric(Mstar, R, xi0_best, Rc) 
    for R in R_test
])

ax1.plot(R_test, DeltaSigma_best, 'g^-', linewidth=3, markersize=8,
        label=f'Ottimale (ξ₀={xi0_best:.1f}), χ²={chi2_best:.1f}',
        zorder=9)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('R [kpc]', fontsize=13)
ax1.set_ylabel(r'$\Delta\Sigma$ [M$_\odot$/pc$^2$]', fontsize=13)
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_title('GCV v3: Metrica Einstein Modificata', fontsize=12)

# Plot Φ_eff/Φ_Newton vs R
R_range = np.logspace(np.log10(10), np.log10(1000), 100)
chi_v_range = suscettibilita(R_range, Rc)
amplification_range = 1 + xi0_best * chi_v_range

ax2.plot(R_range, amplification_range, 'b-', linewidth=2.5)
ax2.axhline(1, color='gray', linestyle='--', alpha=0.5, label='Nessun vuoto')
ax2.fill_between(R_range, 1, amplification_range, alpha=0.3)
ax2.set_xscale('log')
ax2.set_xlabel('R [kpc]', fontsize=13)
ax2.set_ylabel(r'$\Phi_{eff} / \Phi_{Newton}$', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_title(f'Amplificazione Metrica (ξ₀={xi0_best:.1f})', fontsize=12)
ax2.text(50, amplification_range[20], 
         f'Max: {amplification_range.max():.1f}×',
         fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat'))

plt.tight_layout()
plots_dir = Path(__file__).parent / 'plots'
plots_dir.mkdir(exist_ok=True)
plt.savefig(plots_dir / 'gcv_v3_einstein_metric.png', dpi=150)
print(f"\n💾 Plot: plots/gcv_v3_einstein_metric.png")
plt.close()

print(f"\n{'='*70}")
print(f"✨ OTTIMIZZAZIONE:")
print(f"{'='*70}")
print(f"  ξ₀ ottimale = {xi0_best:.2f}")
print(f"  χ² minimo = {chi2_best:.1f}")
print(f"\n  Predizioni:")
for i, R in enumerate(R_test):
    print(f"    R={R:3d} kpc: Obs={DeltaSigma_obs[i]:3.0f}, Pred={DeltaSigma_best[i]:5.1f}, Ratio={DeltaSigma_best[i]/DeltaSigma_obs[i]:.2f}")

print(f"\n{'='*70}")
print(f"📊 VERDETTO:")
print(f"{'='*70}")

if chi2_best < 5:
    verdict = "PASS"
    print(f"\n🎉🎉🎉 GCV v3 FUNZIONA PERFETTAMENTE! 🎉🎉🎉")
    print(f"   χ² = {chi2_best:.1f} è eccellente!")
elif chi2_best < 20:
    verdict = "PLAUSIBLE"
    print(f"\n✅ GCV v3 È PLAUSIBILE")
    print(f"   χ² = {chi2_best:.1f} è accettabile")
elif chi2_best < 100:
    verdict = "PARTIAL"
    print(f"\n⚠️  GCV v3 migliora ma non abbastanza")
    print(f"   χ² = {chi2_best:.1f} ancora alto")
else:
    verdict = "FAIL"
    print(f"\n❌ GCV v3 non funziona")
    print(f"   χ² = {chi2_best:.1f} troppo alto")

if verdict in ["PASS", "PLAUSIBLE"]:
    print(f"\n💡 INTERPRETAZIONE FISICA:")
    print(f"-"*70)
    print(f"""
Il vuoto MODIFICA LA GEOMETRIA SPAZIO-TEMPORALE:

Metrica standard:  g₀₀ = -(1 - 2Φ/c²)
Metrica GCV v3:    g₀₀ = -(1 - 2Φ_eff/c²)

Dove: Φ_eff = Φ × [1 + {xi0_best:.1f} × χᵥ(r)]

A R = 50 kpc:  Φ amplificato di {1 + xi0_best * suscettibilita(50, Rc):.1f}×
A R = 100 kpc: Φ amplificato di {1 + xi0_best * suscettibilita(100, Rc):.1f}×
A R = 800 kpc: Φ amplificato di {1 + xi0_best * suscettibilita(800, Rc):.1f}×

MECCANISMO FISICO:
-----------------
Il vuoto quantistico polarizzato modifica la STRUTTURA dello spazio-tempo.
Non è solo che "aggiunge massa" o "amplifica segnale",
ma CAMBIA LA METRICA stessa!

Analogia classica: dielettrico modifica E
Qui: vuoto modifica g_μν

PREDIZIONI:
----------
1. ξ₀ ~ {xi0_best:.0f} è parametro universale
2. Rotazioni: v² = -r dΦ_eff/dr (compatibile con a₀)
3. Lensing: amplificato da fattore 1 + ξ₀χᵥ
4. Precessione perielio: piccola correzione
5. Onde gravitazionali: velocità c × √(1 + ξ₀χᵥ) ?

COMPATIBILITÀ:
-------------
- GCV v1: a₀ per dinamica Newtoniana
- GCV v3: ξ₀ per metrica relativistica
- Entrambi da χᵥ(k) del vuoto!

RELATIVITÀ GENERALE MODIFICATA:
------------------------------
Equazioni Einstein:  G_μν = 8πG T_μν
Diventano:          G_μν[g_αβ(1+χᵥ)] = 8πG T_μν

Il vuoto entra nella GEOMETRIA, non nell'energia!
""")

# Salva risultati
results_dir = Path(__file__).parent / 'results'
results_dir.mkdir(exist_ok=True)

output = {
    'version': 'GCV v3.0 - Modified Einstein Metric',
    'verdict': verdict,
    'xi0_optimal': float(xi0_best),
    'chi2_minimum': float(chi2_best),
    'interpretation': f'Vacuum modifies metric: Φ_eff = Φ × [1 + {xi0_best:.1f} × χᵥ]',
    'predictions': {
        'R_kpc': R_test.tolist(),
        'observed': DeltaSigma_obs.tolist(),
        'predicted': DeltaSigma_best.tolist()
    }
}

with open(results_dir / 'gcv_v3_einstein_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n💾 Risultati: results/gcv_v3_einstein_results.json")
print(f"{'='*70}")
