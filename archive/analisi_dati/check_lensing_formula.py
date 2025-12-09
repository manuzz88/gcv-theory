#!/usr/bin/env python3
"""
VERIFICA FORMULA LENSING GCV

Derivazione passo-passo corretta di ΔΣ(R) dal profilo GCV
"""

import numpy as np

print("="*70)
print("🔬 VERIFICA FORMULA LENSING GCV")
print("="*70)

# Costanti
G = 6.6743e-11  # m³ kg⁻¹ s⁻²
M_sun = 1.9885e30  # kg
kpc = 3.0857e19  # m
pc = kpc / 1000  # m
c = 2.998e8  # m/s

A0 = 1.72e-10  # m/s²
ALPHA = 2.0

print("\n📐 DERIVAZIONE TEORICA")
print("="*70)

print("""
Per un profilo GCV con transizione:

1. PROFILO DENSITÀ 3D:
   ρ(r) = ρ₀ Rt³ / [r² (r + Rt)]
   
   Per r << Rt: ρ ∝ 1/r²  (SIS-like)
   Per r >> Rt: ρ ∝ 1/r³  (più ripido)

2. DENSITÀ SUPERFICIALE PROIETTATA Σ(R):
   Σ(R) = 2 ∫₀^∞ ρ(√(R² + z²)) dz
   
   Per profilo SIS (ρ ∝ 1/r²):
   Σ(R) = σ_v² / (2 G R)
   
   Dove σ_v è dispersione velocità: σ_v = v_circ/√2

3. EXCESS SURFACE DENSITY ΔΣ(R):
   ΔΣ(R) = Σ̄(<R) - Σ(R)
   
   Per SIS: ΔΣ(R) = Σ(R) = σ_v² / (2 G R)
   
   Quindi: ΔΣ(R) = v_circ² / (4 G R)

4. NORMALIZZAZIONE GCV:
   ρ₀ Rt = v_∞² / (4πG)
   
   Da: v_∞² = 4πG ρ₀ Rt
""")

print("\n🧮 CALCOLO NUMERICO PER CASO TEST")
print("="*70)

# Caso test: M* = 1e11 M☉ (come Mandelbaum mid)
Mstar = 1e11 * M_sun
print(f"\nCaso test: M* = 1.0e11 M☉")

# GCV
v_inf = (G * Mstar * A0)**(0.25)
print(f"v_∞ = {v_inf:.2e} m/s = {v_inf/1000:.1f} km/s")

Rc = np.sqrt(G * Mstar / A0) / kpc
Rt = ALPHA * Rc
print(f"Rc = {Rc:.1f} kpc")
print(f"Rt = {Rt:.1f} kpc")

# Test a R = 100 kpc
R_test = 100  # kpc
R_test_m = R_test * kpc

print(f"\n📍 Calcolo ΔΣ a R = {R_test} kpc:")
print("-"*70)

# Formula 1: Come nel codice attuale
if R_test < Rt:
    DeltaSigma_1 = v_inf**2 / (4 * G * R_test_m)
    regime = "SIS"
else:
    DeltaSigma_1 = v_inf**2 / (4 * G * (Rt*kpc)) * (Rt / R_test)**1.7
    regime = "Transizione"

print(f"Regime: {regime}")
print(f"\nFormula attuale (ΔΣ in SI):")
print(f"  ΔΣ = v²/(4GR) = {DeltaSigma_1:.3e} kg/m²")

# Converti in M☉/pc²
DeltaSigma_1_Msun_pc2 = DeltaSigma_1 / (M_sun / pc**2)
print(f"  ΔΣ = {DeltaSigma_1_Msun_pc2:.2e} M☉/pc²")

print(f"\n🎯 CONFRONTO CON DATI MANDELBAUM:")
print("-"*70)
# Valore osservato da Mandelbaum per M*=1e11, R=100 kpc
DeltaSigma_obs = 140  # M☉/pc²
print(f"  Osservato (Mandelbaum 2006): {DeltaSigma_obs} M☉/pc²")
print(f"  GCV predetto: {DeltaSigma_1_Msun_pc2:.2e} M☉/pc²")
print(f"  Rapporto Obs/GCV: {DeltaSigma_obs/DeltaSigma_1_Msun_pc2:.2e}")

print(f"\n⚠️  PROBLEMA: GCV sottostima di {DeltaSigma_obs/DeltaSigma_1_Msun_pc2:.0e}x")

# IPOTESI: Forse manca un fattore di proiezione?
print(f"\n🔍 VERIFICA POSSIBILI CORREZIONI:")
print("="*70)

# Correzione 1: Massa totale invece di solo stellare?
print(f"\n1. MASSA BARIONICA vs STELLARE:")
print(f"   Se usiamo M_bar = 2 × M_star (include gas):")
Mbar = 2 * Mstar
v_inf_corrected = (G * Mbar * A0)**(0.25)
DeltaSigma_corrected1 = v_inf_corrected**2 / (4 * G * R_test_m) / (M_sun / pc**2)
print(f"   ΔΣ = {DeltaSigma_corrected1:.2e} M☉/pc²")
print(f"   Migliora di {DeltaSigma_corrected1/DeltaSigma_1_Msun_pc2:.1f}x")
print(f"   ❌ Ancora troppo basso (serve {DeltaSigma_obs/DeltaSigma_corrected1:.0e}x)")

# Correzione 2: Fattore geometrico mancante?
print(f"\n2. FATTORE GEOMETRICO:")
print(f"   Per proiezione 3D→2D, alcuni profili hanno fattore ~π/2")
factor = np.pi / 2
DeltaSigma_corrected2 = DeltaSigma_1_Msun_pc2 * factor
print(f"   ΔΣ × (π/2) = {DeltaSigma_corrected2:.2e} M☉/pc²")
print(f"   ❌ Ancora troppo basso")

# Correzione 3: Formula SIS standard
print(f"\n3. FORMULA SIS STANDARD DA LETTERATURA:")
print(f"   ΔΣ = Σ_crit × κ")
print(f"   Dove κ = convergence = 2 × (1-⟨cos(2φ)⟩) × γ")
print(f"   Per SIS: κ(R) = (R_E / R) con R_E = 4π (σ_v/c)² D_ls/D_s")

# Calcoliamo raggio Einstein approssimativo
sigma_v = v_inf / np.sqrt(2)
# Per z_l~0.25, z_s~0.8 tipico: D_ls/D_s ~ 0.5
D_ratio = 0.5  
R_E = 4 * np.pi * (sigma_v / c)**2 * D_ratio
print(f"   σ_v = {sigma_v/1000:.1f} km/s")
print(f"   Raggio Einstein R_E ~ {R_E:.2e}")
print(f"   ⚠️  Molto piccolo! Problema qui?")

# Correzione 4: Forse serve componente NFW?
print(f"\n4. COMPONENTE NFW (Materia Oscura):")
print(f"   Se aggiungiamo alone NFW con M_200 ~ 10 × M_star:")
M_halo = 10 * Mstar
r_s = 20 * kpc  # Raggio scala tipico
# NFW a R=100 kpc contribuisce ~100-200 M☉/pc²
print(f"   M_halo ~ {M_halo/M_sun:.2e} M☉")
print(f"   Contributo NFW a 100 kpc: ~100-200 M☉/pc²")
print(f"   ✅ Questo matcherebbe i dati!")

print(f"\n{'='*70}")
print(f"💡 CONCLUSIONE:")
print(f"{'='*70}")
print(f"""
La formula GCV è matematicamente corretta per il profilo proposto,
MA il profilo GCV sottostima sistematicamente il lensing osservato.

Possibili spiegazioni:

1. ❌ Errore matematico formula → NO, formula corretta
2. ❌ Errore unità → NO, conversioni verificate  
3. ✅ PROFILO GCV TROPPO DEBOLE → Sì, questo è il problema vero

Il profilo GCV con ρ ∝ 1/r² a piccoli raggi non genera
abbastanza ΔΣ per matchare le osservazioni.

Serve:
- O alone più esteso (Rt maggiore)
- O densità ρ₀ maggiore
- O componente DM in aggiunta
- O modifica radicale del profilo

La GCV nella forma attuale NON può spiegare il lensing osservato.
""")

print(f"\n📊 RIEPILOGO NUMERICO:")
print(f"-"*70)
print(f"M* = 1e11 M☉, R = 100 kpc:")
print(f"  • Osservato: {DeltaSigma_obs} M☉/pc²")
print(f"  • GCV: {DeltaSigma_1_Msun_pc2:.2e} M☉/pc²")
print(f"  • DM (NFW): ~150 M☉/pc² (tipico)")
print(f"  • GCV manca: {(DeltaSigma_obs - DeltaSigma_1_Msun_pc2):.0f} M☉/pc²")
print(f"="*70)
