#!/usr/bin/env python3
"""
GCV v2.0: MECCANISMO CURVATURA RIVISTO

Problema GCV v1: Modifica Poisson ma non genera abbastanza curvatura
Soluzione GCV v2: Il vuoto contribuisce direttamente al tensore energia-momento

APPROCCIO:
----------
Invece di: ∇·[(1 + χᵥ)∇Φ] = 4πG ρ_b (Poisson modificato)

Usiamo Einstein completo:
  Gμν = 8πG (T_matter + T_vacuum)

Dove T_vacuum emerge dalla risposta del vuoto al campo gravitazionale
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("="*70)
print("🔬 GCV v2.0: MECCANISMO CURVATURA")
print("="*70)

print("""
PROBLEMA IDENTIFICATO:
---------------------
GCV v1 modifica solo il POTENZIALE Φ tramite χᵥ
Ma il lensing dipende dalla CURVATURA dello spazio-tempo

Per Einstein: curvatura = densità energia-momento

Se χᵥ modifica Φ, DEVE anche modificare la metrica!

SOLUZIONE PROPOSTA:
------------------
Il vuoto con suscettibilità χᵥ ha una densità energetica indotta:

  ρ_vac(r) = f(χᵥ, ρ_b, Φ)

Questa densità contribuisce DIRETTAMENTE alla curvatura:
  
  Gμν = 8πG (ρ_b + ρ_vac) × (velocity terms)

CANDIDATI PER ρ_vac:
-------------------

1. POLARIZZAZIONE LINEARE:
   ρ_vac = χᵥ × ρ_b
   
   Pro: Semplice, già implicitamente nella GCV
   Contro: Non basta (già testato implicitamente)

2. GRADIENTE QUADRATO (DIELETTRICO):
   ρ_vac = χᵥ × (∇Φ)² / (8πG c²)
   
   Pro: Analogo a energia dielettrica E²/2
   Contro: Testato, contributo trascurabile

3. DENSITÀ INDOTTA PROPORZIONALE A CURVATURA:
   ρ_vac = χᵥ × R × (c²/G)
   
   Dove R = scalare di Ricci
   Pro: Self-consistent, R dipende da ρ_tot
   Contro: Equazione implicita, complessa

4. STRESS ANISOTROPO (EFFETTO TIDAL):
   T_vac ha componenti NON-DIAGONALI
   
   Pro: Può generare lensing extra senza massa
   Contro: Difficile da calcolare

5. **EMERGENT MASS dal VUOTO (CANDIDATO PRINCIPALE)**:
   
   M_eff(r) = M_b × [1 + η × χᵥ(r)]
   
   Dove η è coefficiente di accoppiamento vuoto-materia
   
   Pro: Semplice, mantiene struttura equazioni
   Pro: η è NUOVO parametro per lensing
   Contro: Serve giustificare fisicamente

IMPLEMENTAZIONE PROPOSTA (Opzione 5):
------------------------------------
""")

# Costanti
G = 6.6743e-11
M_sun = 1.9885e30
kpc = 3.0857e19
pc = kpc / 1000
c = 2.998e8

A0 = 1.72e-10
ALPHA = 2.0

print("\n🧮 TEST MECCANISMO: MASSA EFFETTIVA")
print("="*70)

print("""
Ipotesi: Il vuoto "traveste" la materia aumentandone massa apparente

M_eff(r) = M_b × [1 + η(r)]

Dove η(r) = η₀ × χᵥ(r)

Per r << Lc: χᵥ → χ₀ → η grande → M_eff aumenta
Per r >> Lc: χᵥ → 0   → η → 0  → M_eff = M_b

Questo genera:
- Rotazioni: v² = G M_eff(r) / r con M_eff > M_b → curve piatte ✓
- Lensing: ΔΣ ∝ M_eff proiettato → segnale maggiore ✓
""")

# Parametri galassia test
Mstar = 1e11 * M_sun
v_inf = (G * Mstar * A0)**(0.25)
Rc = np.sqrt(G * Mstar / A0) / kpc
Rt = ALPHA * Rc

print(f"\nGalassia: M* = 1e11 M☉")
print(f"  Rc = {Rc:.1f} kpc")

# Test vari valori di η₀
R_test = np.array([50, 100, 200, 400, 800])  # kpc
DeltaSigma_obs = np.array([200, 140, 80, 35, 15])  # M☉/pc²

print(f"\n📊 TEST DIVERSI η₀:")
print("-"*70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for eta0 in [0, 2, 5, 10, 20]:
    DeltaSigma_pred = []
    
    for R in R_test:
        # Suscettibilità a raggio R
        k = 1 / R  # kpc⁻¹
        Lc = Rc
        chi_v = 1 / (1 + (k * Lc)**2)
        
        # Massa effettiva
        eta_R = eta0 * chi_v
        M_eff = Mstar * (1 + eta_R)
        
        # ΔΣ con massa effettiva
        # Semplificazione: ΔΣ ∝ M_eff / R
        v_eff = (G * M_eff * M_sun * A0)**(0.25)
        Rt_eff = ALPHA * np.sqrt(G * M_eff * M_sun / A0) / kpc
        
        R_m = R * kpc
        if R < Rt_eff:
            ds = v_eff**2 / (4 * G * R_m)
        else:
            ds = v_eff**2 / (4 * G * (Rt_eff*kpc)) * (Rt_eff / R)**1.7
        
        DeltaSigma_pred.append(ds / (M_sun / pc**2))
    
    DeltaSigma_pred = np.array(DeltaSigma_pred)
    
    # χ² con osservazioni
    chi2 = np.sum((DeltaSigma_obs - DeltaSigma_pred)**2 / DeltaSigma_obs)
    
    # Plot
    ax1.plot(R_test, DeltaSigma_pred, 'o-', label=f'η₀={eta0}, χ²={chi2:.0f}', 
            linewidth=2, markersize=6)
    
    print(f"  η₀ = {eta0:2d}: χ² = {chi2:6.1f}, ΔΣ(100kpc) = {DeltaSigma_pred[1]:6.1f} M☉/pc²")

# Osservato
ax1.plot(R_test, DeltaSigma_obs, 'ks-', linewidth=3, markersize=10,
        label='Osservato', zorder=10)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('R [kpc]', fontsize=12)
ax1.set_ylabel(r'$\Delta\Sigma$ [M$_\odot$/pc$^2$]', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_title('GCV v2: Massa Effettiva dal Vuoto', fontsize=12)

# Ottimizzazione η₀
from scipy.optimize import minimize

def chi2_func(eta0):
    DeltaSigma_pred = []
    for R in R_test:
        k = 1 / R
        chi_v = 1 / (1 + (k * Rc)**2)
        eta_R = eta0[0] * chi_v
        M_eff = Mstar * (1 + eta_R)
        
        v_eff = (G * M_eff * M_sun * A0)**(0.25)
        Rt_eff = ALPHA * np.sqrt(G * M_eff * M_sun / A0) / kpc
        
        R_m = R * kpc
        if R < Rt_eff:
            ds = v_eff**2 / (4 * G * R_m)
        else:
            ds = v_eff**2 / (4 * G * (Rt_eff*kpc)) * (Rt_eff / R)**1.7
        
        DeltaSigma_pred.append(ds / (M_sun / pc**2))
    
    DeltaSigma_pred = np.array(DeltaSigma_pred)
    return np.sum((DeltaSigma_obs - DeltaSigma_pred)**2 / DeltaSigma_obs)

result = minimize(chi2_func, [10], bounds=[(0, 100)])
eta_best = result.x[0]
chi2_best = result.fun

print(f"\n✨ OTTIMIZZAZIONE:")
print(f"  η₀ ottimale = {eta_best:.2f}")
print(f"  χ² minimo = {chi2_best:.1f}")

# Plot con η ottimale
DeltaSigma_best = []
for R in R_test:
    k = 1 / R
    chi_v = 1 / (1 + (k * Rc)**2)
    eta_R = eta_best * chi_v
    M_eff = Mstar * (1 + eta_R)
    
    v_eff = (G * M_eff * M_sun * A0)**(0.25)
    Rt_eff = ALPHA * np.sqrt(G * M_eff * M_sun / A0) / kpc
    
    R_m = R * kpc
    if R < Rt_eff:
        ds = v_eff**2 / (4 * G * R_m)
    else:
        ds = v_eff**2 / (4 * G * (Rt_eff*kpc)) * (Rt_eff / R)**1.7
    
    DeltaSigma_best.append(ds / (M_sun / pc**2))

DeltaSigma_best = np.array(DeltaSigma_best)

ax1.plot(R_test, DeltaSigma_best, 'g^-', linewidth=3, markersize=8,
        label=f'Ottimale (η₀={eta_best:.1f})', zorder=9)
ax1.legend(fontsize=9)

# Plot η(R) vs χᵥ(R)
R_range = np.logspace(np.log10(30), np.log10(1000), 50)
chi_v_range = 1 / (1 + (1/R_range * Rc)**2)
eta_range = eta_best * chi_v_range

ax2.plot(R_range, chi_v_range, 'b-', linewidth=2, label=r'$\chi_v(R)$')
ax2.plot(R_range, eta_range, 'r-', linewidth=2, label=r'$\eta(R) = \eta_0 \chi_v$')
ax2.axhline(eta_best, color='gray', linestyle='--', alpha=0.5)
ax2.set_xscale('log')
ax2.set_xlabel('R [kpc]', fontsize=12)
ax2.set_ylabel('Fattore', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_title(f'Profilo η(R) con η₀={eta_best:.1f}', fontsize=12)

plt.tight_layout()
plots_dir = Path(__file__).parent / 'plots'
plots_dir.mkdir(exist_ok=True)
plt.savefig(plots_dir / 'gcv_v2_effective_mass.png', dpi=150)
print(f"\n💾 Plot: plots/gcv_v2_effective_mass.png")
plt.close()

print(f"\n{'='*70}")
print(f"💡 INTERPRETAZIONE FISICA:")
print(f"{'='*70}")

print(f"""
Con η₀ = {eta_best:.1f}, il vuoto "amplifica" la massa apparente:

A piccoli raggi (R ~ Rc ~ {Rc:.0f} kpc):
  χᵥ ~ 1
  η ~ {eta_best:.1f}
  M_eff ~ {1+eta_best:.1f} × M_b
  
  → Il vuoto rende la galassia {1+eta_best:.1f}x più "pesante" per il lensing!

A grandi raggi (R >> Rc):
  χᵥ → 0
  η → 0
  M_eff → M_b
  
  → Torna massa normale

MECCANISMO FISICO POSSIBILE:
----------------------------
Il vuoto quantistico, polarizzato dal campo gravitazionale,
crea una "nuvola" di coppie virtuali che amplificano l'effetto
gravitazionale della materia. 

Analogia: dielettrico amplifica campo elettrico
Qui: vuoto amplifica campo gravitazionale

PREDIZIONI TESTABILI:
--------------------
1. η₀ è UNIVERSALE (stesso per tutte le galassie)
2. η₀ ~ {eta_best:.0f} richiede forte accoppiamento vuoto-gravità
3. Lc determina dove η diventa efficace
4. Su rotazioni: effetto già catturato da a₀
5. Su lensing: effetto AGGIUNTIVO tramite η

COMPATIBILITÀ CON GCV v1:
------------------------
GCV v1 con a₀ → spiega rotazioni
GCV v2 con η₀ → spiega lensing
Entrambi emergono dallo stesso χᵥ!

a₀ = parametro dinamico (equazioni moto)
η₀ = parametro geometrico (curvatura spazio)
""")

if chi2_best < 20:
    print(f"\n🎉🎉🎉 GCV v2 CON η₀={eta_best:.1f} PUÒ FUNZIONARE! 🎉🎉🎉")
    print(f"   χ² = {chi2_best:.1f} è accettabile")
    print(f"   Serve testare su tutti i dataset")
elif chi2_best < 100:
    print(f"\n⚠️  GCV v2 migliora ma non è perfetta")
    print(f"   χ² = {chi2_best:.1f} ancora alto")
    print(f"   Serve raffinare meccanismo")
else:
    print(f"\n❌ GCV v2 non basta")
    print(f"   χ² = {chi2_best:.1f} troppo alto")

print(f"{'='*70}")
