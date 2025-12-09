#!/usr/bin/env python3
"""
PEZZO MANCANTE: ENERGIA DEL VUOTO ATTIVO

Se χᵥ > 0, il vuoto ha energia che contribuisce alla curvatura!
Questa energia crea lensing AGGIUNTIVO oltre alla materia.
"""

import numpy as np

print("="*70)
print("🧩 PEZZO MANCANTE: DENSITÀ ENERGETICA DEL VUOTO")
print("="*70)

# Costanti
G = 6.6743e-11  # m³ kg⁻¹ s⁻²
M_sun = 1.9885e30  # kg
kpc = 3.0857e19  # m
pc = kpc / 1000  # m
c = 2.998e8  # m/s

A0 = 1.72e-10  # m/s²
ALPHA = 2.0

print("""
IPOTESI:
--------
La suscettibilità χᵥ implica che il vuoto ha energia.

Se ∇·[(1 + χᵥ)∇Φ] = 4πG ρb, allora possiamo scrivere:
∇²Φ = 4πG (ρb + ρ_vacuum)

Dove: ρ_vacuum = (χᵥ/(1+χᵥ)) × ρb × (qualche fattore)

Ma meglio: il vuoto con χᵥ ha un'energia di "polarizzazione":

ε_vacuum ~ χᵥ × (∇Φ)² / (8πG)

Questo è analogo all'energia di un dielettrico: ε = ε₀ E² / 2
""")

print("\n🧮 STIMA ORDINE DI GRANDEZZA")
print("="*70)

# Caso test: M* = 1e11 M☉
Mstar = 1e11 * M_sun
v_inf = (G * Mstar * A0)**(0.25)
Rc = np.sqrt(G * Mstar / A0) / kpc
Rt = ALPHA * Rc

print(f"\nParametri galassia: M* = 1e11 M☉")
print(f"  v_∞ = {v_inf/1000:.1f} km/s")
print(f"  Rc = {Rc:.1f} kpc")
print(f"  Rt = {Rt:.1f} kpc")

# Stima χᵥ(k) a scala R ~ 100 kpc
R_test = 100  # kpc
k_test = 1 / R_test  # kpc⁻¹

# Assumiamo Lc ~ Rc
Lc = Rc
chi_v = 1 / (1 + (k_test * Lc)**2)

print(f"\n📐 Suscettibilità a R = {R_test} kpc:")
print(f"  k = 1/R = {k_test:.4f} kpc⁻¹")
print(f"  Lc ~ Rc = {Lc:.1f} kpc")
print(f"  χᵥ(k) = {chi_v:.3f}")

# Gradiente potenziale
# |∇Φ| ~ GM/R² ~ v²/R
grad_Phi = v_inf**2 / (Rt * kpc)

print(f"\n📐 Gradiente potenziale a Rt:")
print(f"  |∇Φ| ~ {grad_Phi:.3e} s⁻²")

# Densità energia vuoto (formula fenomenologica)
# ρ_vac ~ χᵥ × (∇Φ)² / (4πG c²)
rho_vacuum = chi_v * grad_Phi**2 / (4 * np.pi * G * c**2)

print(f"\n⚡ Densità energia vuoto:")
print(f"  ρ_vacuum ~ {rho_vacuum:.3e} kg/m³")

# Confronto con densità materia
rho_matter_avg = Mstar / (4/3 * np.pi * (Rt * kpc)**3)
print(f"  ρ_matter ~ {rho_matter_avg:.3e} kg/m³ (media entro Rt)")
print(f"  Rapporto ρ_vac/ρ_matter = {rho_vacuum/rho_matter_avg:.2e}")

# Contributo al lensing
# Se ρ_vacuum contribuisce, ΔΣ aumenta
R_test_m = R_test * kpc

# ΔΣ solo materia (quello che abbiamo ora)
DeltaSigma_matter = v_inf**2 / (4 * G * R_test_m) / (M_sun / pc**2)

# ΔΣ dal vuoto: ipotesi che ρ_vacuum integrato dia contributo simile
# Integrale ∫ ρ_vacuum dz lungo linea vista
# Approssimazione: contributo ~ ρ_vacuum × 2Rt
M_vacuum_eff = rho_vacuum * 4 * np.pi * (Rt * kpc)**2 * (2 * Rt * kpc)
DeltaSigma_vacuum = M_vacuum_eff / (np.pi * (R_test * pc)**2) / M_sun

print(f"\n📊 CONTRIBUTI AL LENSING a R = {R_test} kpc:")
print(f"  ΔΣ_matter = {DeltaSigma_matter:.1f} M☉/pc² (solo barionica)")
print(f"  ΔΣ_vacuum = {DeltaSigma_vacuum:.1f} M☉/pc² (energia vuoto)")
print(f"  ΔΣ_TOTALE = {DeltaSigma_matter + DeltaSigma_vacuum:.1f} M☉/pc²")

# Confronto con osservato
DeltaSigma_obs = 140
print(f"\n  Osservato: {DeltaSigma_obs} M☉/pc²")
print(f"  GCV solo materia: {DeltaSigma_matter:.1f} M☉/pc²")
print(f"  GCV con vuoto: {DeltaSigma_matter + DeltaSigma_vacuum:.1f} M☉/pc²")

factor_improvement = (DeltaSigma_matter + DeltaSigma_vacuum) / DeltaSigma_matter
print(f"\n  ✨ Miglioramento: {factor_improvement:.1f}x")

# Quanto manca ancora?
missing = DeltaSigma_obs - (DeltaSigma_matter + DeltaSigma_vacuum)
print(f"  Manca ancora: {missing:.1f} M☉/pc²")

if missing < DeltaSigma_obs * 0.3:
    print(f"  ✅ Entro 30% - PLAUSIBILE!")
elif missing < DeltaSigma_obs * 0.5:
    print(f"  ⚠️  Entro 50% - Serve raffinare")
else:
    print(f"  ❌ Oltre 50% - Non basta")

print(f"\n{'='*70}")
print(f"💡 ALTRE POSSIBILITÀ:")
print(f"{'='*70}")

print("""
2. SCALA DI COERENZA DIPENDENTE DA SCALA DI OSSERVAZIONE:
   - Sul lensing (scale ~Mpc) forse Lc è diverso
   - Lc = Lc(k, Σb) invece di solo Lc(Mb)?
   - Potrebbe dare χᵥ più alto su grandi scale

3. EFFETTO INTEGRATO LUNGO LINEA DI VISTA:
   - Il lensing integra da z_lens a z_source
   - Il vuoto accumula effetti su ~100 Mpc
   - Potrebbe amplificare oltre la semplice proiezione

4. TERMINE COSMOLOGICO NEL VUOTO:
   - χᵥ potrebbe dipendere da H(z)
   - A z più alto, vuoto più attivo?
   - Effetto evoluzione cosmologica

5. CONTRIBUTO QUADRATICO:
   - Forse (∇Φ)² dà contributo non-lineare
   - Termini di ordine superiore in χᵥ
   - Auto-interazione del campo di vuoto

6. DIPENDENZA DA AMBIENTE:
   - In regioni dense (dentro galassie) χᵥ saturato?
   - In vuoti cosmici χᵥ più forte?
   - Effetto non-locale da struttura circostante
""")

print(f"\n{'='*70}")
print(f"🎯 RACCOMANDAZIONI:")
print(f"{'='*70}")
print("""
1. CALCOLA PROPRIAMENTE ρ_vacuum da teoria campo
   - Deriva da Lagrangiana con χᵥ
   - Includi nel tensore energia-momento
   - Risolvi equazioni Einstein complete

2. TESTA Lc DIPENDENTE DA SCALA:
   - Lc_lensing > Lc_rotations?
   - Forse Lc ~ √(R × Rc)?
   - Fit su 2 scale diverse

3. INTEGRALE COSMOLOGICO:
   - Calcola ∫ ρ_vacuum(z) dz lungo vista
   - Include evoluzione H(z)
   - Effetto cumulativo

4. NEAR-FIELD vs FAR-FIELD:
   - Rotazioni: campo near (dentro galassia)
   - Lensing: campo far (proiezione Mpc)
   - Comportamento χᵥ potrebbe essere diverso!
""")

print(f"{'='*70}")
