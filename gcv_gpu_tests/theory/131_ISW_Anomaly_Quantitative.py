#!/usr/bin/env python3
"""
GCV UNIFIED: ISW ANOMALY QUANTITATIVE TEST
============================================

Script 131 - February 2026

Granett, Neyrinck & Szapudi (2008) found:
  ISW signal from supervoids ~2× stronger than LCDM predicts
  ΔT ~ -11.3 ± 3.1 μK from 50 supervoids (SDSS DR6)

This has been a PERSISTENT ANOMALY in LCDM cosmology.

GCV PREDICTS enhanced ISW from voids because chi_v < 1 in
underdense regions → potentials decay FASTER.

QUESTION: Does GCV predict exactly ~2× enhancement?

Author: Manuel Lazzaro
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# =============================================================================
# CONSTANTS
# =============================================================================

G = 6.674e-11
c = 2.998e8
H0_si = 2.184e-18
H0_km = 67.4
Mpc = 3.086e22

Omega_m = 0.315
Omega_Lambda = 0.685
rho_crit_0 = 3 * H0_si**2 / (8 * np.pi * G)
rho_t = Omega_Lambda * rho_crit_0
a0 = 1.2e-10
T_cmb = 2.7255  # K

chi_vacuum = 1 - Omega_Lambda / Omega_m

print("=" * 75)
print("SCRIPT 131: ISW ANOMALY QUANTITATIVE TEST")
print("=" * 75)

# =============================================================================
# PART 1: THE ISW EFFECT IN LCDM
# =============================================================================

print("\n" + "=" * 75)
print("PART 1: ISW EFFECT PHYSICS")
print("=" * 75)

print("""
THE ISW EFFECT:
  A photon traversing a time-varying gravitational potential gains/loses energy:
    ΔT/T = (2/c³) ∫ (∂Φ/∂t) dl

  In LCDM, Φ decays at late times because dark energy suppresses growth:
    ∂Φ/∂t = Φ₀ * [d/dt(D(t)/a(t))]

  For a void of radius R_v, central underdensity δ_v at redshift z_v:
    ΔT_ISW ≈ -(2/3) * (Ω_m/Ω_Λ) * (H₀R_v/c)² * δ_v * D_dot/(aH) * T_CMB

OBSERVED (Granett+2008):
  50 supervoids at z ~ 0.5, R ~ 100 Mpc/h
  ΔT_observed = -11.3 ± 3.1 μK (stacked)
  LCDM prediction: ΔT_LCDM ~ -5 to -7 μK
  
  → ANOMALY: observed/predicted ~ 1.7-2.3×
""")

# =============================================================================
# PART 2: ISW CALCULATION
# =============================================================================

print("\n" + "=" * 75)
print("PART 2: ISW SIGNAL CALCULATION")
print("=" * 75)

def growth_factor(z):
    """Linear growth factor D(z) normalized to D(0) = 1."""
    a = 1 / (1 + z)
    # Approximate Carroll+1992 formula
    omega_z = Omega_m * (1 + z)**3 / (Omega_m * (1 + z)**3 + Omega_Lambda)
    lambda_z = Omega_Lambda / (Omega_m * (1 + z)**3 + Omega_Lambda)
    D = (5/2) * omega_z / (omega_z**(4/7) - lambda_z + (1 + omega_z/2) * (1 + lambda_z/70))
    # Normalize
    omega_0 = Omega_m
    lambda_0 = Omega_Lambda
    D0 = (5/2) * omega_0 / (omega_0**(4/7) - lambda_0 + (1 + omega_0/2) * (1 + lambda_0/70))
    return D / D0

def growth_rate(z):
    """Growth rate f = d ln D / d ln a."""
    omega_z = Omega_m * (1 + z)**3 / (Omega_m * (1 + z)**3 + Omega_Lambda)
    return omega_z**0.55  # Approximate

def hubble(z):
    """Hubble parameter H(z) in s^-1."""
    return H0_si * np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)

def isw_void_lcdm(R_v_Mpc, delta_v, z_v):
    """
    ISW temperature shift from a void in LCDM.
    
    Based on Rees-Sciama / ISW for a compensated void.
    Uses the Inoue & Silk (2006) / Granett+2008 formalism.
    
    R_v_Mpc: void radius in Mpc
    delta_v: central underdensity (negative)
    z_v: void redshift
    """
    R_v = R_v_Mpc * Mpc  # meters
    
    # The ISW signal for a top-hat void:
    # ΔT/T = -(4/3) * π * (R_v/c)³ * G * ρ_bar * δ_v * [f(z) - 1] * (1+z) / c
    # But more accurately using the potential decay rate:
    
    D_z = growth_factor(z_v)
    f_z = growth_rate(z_v)
    a_z = 1 / (1 + z_v)
    H_z = hubble(z_v)
    
    # Potential for a void: Φ = -(4π/3) G ρ_bar δ R_v²
    # With δ < 0: Φ > 0 (potential hill, NOT well)
    rho_bar = Omega_m * rho_crit_0 * (1 + z_v)**3
    Phi_v = -(4.0/3) * np.pi * G * rho_bar * delta_v * R_v**2  # >0 for void
    
    # Rate of potential change:
    # Φ ∝ D(t)/a(t), so dΦ/dt = Φ * H * (f - 1)
    # At late times f < 1, so (f-1) < 0 → dΦ/dt < 0 (hill decays)
    dPhi_dt = Phi_v * (f_z - 1) * H_z
    
    # ISW signal: ΔT/T = (2/c²) ∫ dΦ/dt dl/c ≈ (2/c²) * dΦ/dt * (2R_v/c)
    # For void: dΦ/dt < 0 → ΔT < 0 (cold spot) ✔
    # Apply a geometric filter factor ~0.3 for realistic void profiles
    filter_factor = 0.33  # Accounts for non-top-hat profile and projection
    delta_T_over_T = 2.0 / c**2 * dPhi_dt * (2 * R_v / c) * filter_factor
    delta_T = delta_T_over_T * T_cmb
    
    return delta_T * 1e6  # in μK


def isw_void_gcv(R_v_Mpc, delta_v, z_v):
    """
    ISW temperature shift from a void in GCV Unified.
    
    In GCV, the potential decays FASTER in voids because:
    1. Standard decay from DE (same as LCDM)
    2. Additional decay because Gamma < 1 in underdense regions
       means the effective gravitational coupling is evolving
    
    The enhancement is modest (1.3-2.0×) because:
    - Cosmic voids are still at rho ~ 0.3-0.7 rho_bar
    - Gamma(rho_void) ~ 0.5-0.9 (partial transition)
    - The chi_v evolution adds ~30-100% to the decay rate
    """
    R_v = R_v_Mpc * Mpc
    
    D_z = growth_factor(z_v)
    f_z = growth_rate(z_v)
    H_z = hubble(z_v)
    
    rho_bar = Omega_m * rho_crit_0 * (1 + z_v)**3
    rho_void = rho_bar * (1 + delta_v)
    rho_void = max(rho_void, 1e-35)
    
    # Gamma inside void vs background
    gamma_void = np.tanh(rho_void / rho_t)
    gamma_bg = np.tanh(rho_bar / rho_t)
    
    # The GCV potential for the void:
    Phi_v = -(4.0/3) * np.pi * G * rho_bar * delta_v * R_v**2  # >0 for void
    
    # Standard ISW rate (from growth factor decay)
    dPhi_dt_standard = Phi_v * (f_z - 1) * H_z  # <0 for void
    
    # GCV ENHANCEMENT:
    # In GCV, the gravitational potential is modulated by chi_v.
    # The DIFFERENCE in chi_v between void interior and surroundings
    # means the potential profile evolves as Gamma evolves.
    # 
    # The additional decay rate comes from:
    # d/dt[Gamma(rho)] = dGamma/drho * drho/dt
    # As the void expands, rho_void decreases, Gamma decreases,
    # and the effective potential weakens ADDITIONALLY.
    #
    # Enhancement factor:
    # eta = 1 + (Gamma_bg - Gamma_void) / |f - 1|
    # This is bounded and physical.
    
    delta_gamma = gamma_bg - gamma_void  # >0 (bg is denser)
    
    # The enhancement scales with how much Gamma differs
    # between the void and its surroundings.
    # Normalized by the standard ISW driver |f-1|:
    eta = 1.0 + delta_gamma / max(abs(f_z - 1), 0.01)
    
    # Cap the enhancement to be physical (max ~3x)
    eta = min(eta, 3.0)
    
    dPhi_dt_gcv = dPhi_dt_standard * eta
    
    filter_factor = 0.33
    delta_T_over_T = 2.0 / c**2 * dPhi_dt_gcv * (2 * R_v / c) * filter_factor
    delta_T = delta_T_over_T * T_cmb
    
    return delta_T * 1e6, eta  # μK, enhancement factor


# =============================================================================
# PART 3: COMPARISON WITH GRANETT+2008
# =============================================================================

print("\n" + "=" * 75)
print("PART 3: COMPARISON WITH GRANETT+2008 OBSERVATIONS")
print("=" * 75)

# Granett+2008 supervoid parameters
R_void_granett = 100  # Mpc/h → ~140 Mpc
z_void_granett = 0.5
delta_void_granett = -0.4  # Typical supervoid underdensity

# Convert Mpc/h to Mpc
h = H0_km / 100
R_void_Mpc = R_void_granett / h

print(f"\nGranett+2008 supervoid parameters:")
print(f"  R_void = {R_void_granett} Mpc/h = {R_void_Mpc:.0f} Mpc")
print(f"  z_void = {z_void_granett}")
print(f"  delta_void ~ {delta_void_granett}")

# Observed
delta_T_obs = -11.3  # μK
delta_T_obs_err = 3.1  # μK

# LCDM prediction
delta_T_lcdm = isw_void_lcdm(R_void_Mpc, delta_void_granett, z_void_granett)

# GCV prediction
delta_T_gcv, enhancement = isw_void_gcv(R_void_Mpc, delta_void_granett, z_void_granett)

print(f"\nResults:")
print(f"  Observed:  ΔT = {delta_T_obs:.1f} ± {delta_T_obs_err:.1f} μK")
print(f"  LCDM:      ΔT = {delta_T_lcdm:.2f} μK")
print(f"  GCV:       ΔT = {delta_T_gcv:.2f} μK")
print(f"  GCV/LCDM enhancement: {enhancement:.2f}×")
print(f"  Obs/LCDM ratio: {abs(delta_T_obs/delta_T_lcdm):.2f}×")

# Check if GCV is closer to observations
residual_lcdm = abs(delta_T_obs - delta_T_lcdm) / delta_T_obs_err
residual_gcv = abs(delta_T_obs - delta_T_gcv) / delta_T_obs_err

print(f"\n  LCDM residual: {residual_lcdm:.1f}σ from observation")
print(f"  GCV residual:  {residual_gcv:.1f}σ from observation")

if residual_gcv < residual_lcdm:
    print(f"\n  ✅ GCV is CLOSER to the observation than LCDM!")
else:
    print(f"\n  ⚠️ LCDM is closer (but GCV still in right direction)")

# =============================================================================
# PART 4: SCAN OVER VOID PARAMETERS
# =============================================================================

print("\n" + "=" * 75)
print("PART 4: ISW SIGNAL OVER VOID PARAMETER SPACE")
print("=" * 75)

delta_range = np.linspace(-0.9, -0.1, 20)
R_range = np.array([30, 50, 80, 100, 120, 150])  # Mpc/h
z_voids = [0.3, 0.5, 0.7]

print(f"\nEnhancement factor (GCV/LCDM) for different void parameters:")
print(f"\n  At z = 0.5:")
print(f"  {'delta_v':>8}", end='')
for R in R_range:
    print(f" {'R='+str(R):>8}", end='')
print()
print("-" * (10 + 9 * len(R_range)))

for dv in [-0.3, -0.5, -0.7, -0.9]:
    print(f"  {dv:>8.1f}", end='')
    for R in R_range:
        R_Mpc = R / h
        _, enh = isw_void_gcv(R_Mpc, dv, 0.5)
        print(f" {enh:>8.2f}", end='')
    print()

# =============================================================================
# PART 5: STACKED ISW SIGNAL
# =============================================================================

print("\n" + "=" * 75)
print("PART 5: STACKED SIGNAL FOR 50 SUPERVOIDS")
print("=" * 75)

# Simulate 50 supervoids with realistic distribution
np.random.seed(42)
N_voids = 50
R_voids = np.random.normal(100, 20, N_voids) / h  # Mpc
z_voids_sample = np.random.uniform(0.3, 0.7, N_voids)
delta_voids = np.random.uniform(-0.6, -0.2, N_voids)

stacked_lcdm = 0
stacked_gcv = 0
enhancements = []

for i in range(N_voids):
    t_lcdm = isw_void_lcdm(R_voids[i], delta_voids[i], z_voids_sample[i])
    t_gcv, enh = isw_void_gcv(R_voids[i], delta_voids[i], z_voids_sample[i])
    stacked_lcdm += t_lcdm
    stacked_gcv += t_gcv
    enhancements.append(enh)

stacked_lcdm /= N_voids
stacked_gcv /= N_voids
mean_enhancement = np.mean(enhancements)

print(f"Stacked signal from {N_voids} supervoids:")
print(f"  LCDM:     ΔT_stacked = {stacked_lcdm:.2f} μK")
print(f"  GCV:      ΔT_stacked = {stacked_gcv:.2f} μK")
print(f"  Observed: ΔT_stacked = {delta_T_obs:.1f} ± {delta_T_obs_err:.1f} μK")
print(f"  Mean enhancement: {mean_enhancement:.2f}×")

# =============================================================================
# GENERATE PLOTS
# =============================================================================

print("\n\nGenerating figures...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('GCV Unified: ISW Anomaly Quantitative Test (Script 131)',
             fontsize=15, fontweight='bold')

# Plot 1: ISW signal comparison
ax = axes[0, 0]
R_scan = np.linspace(20, 200, 100) / h
isw_lcdm_scan = np.array([isw_void_lcdm(R, -0.4, 0.5) for R in R_scan])
isw_gcv_scan = np.array([isw_void_gcv(R, -0.4, 0.5)[0] for R in R_scan])

ax.plot(R_scan * h, isw_lcdm_scan, 'r--', linewidth=2, label='LCDM')
ax.plot(R_scan * h, isw_gcv_scan, 'b-', linewidth=2.5, label='GCV Unified')
ax.axhline(y=delta_T_obs, color='green', linestyle=':', linewidth=2)
ax.fill_between(R_scan * h, delta_T_obs - delta_T_obs_err, delta_T_obs + delta_T_obs_err,
                alpha=0.2, color='green', label=f'Observed: {delta_T_obs}±{delta_T_obs_err} μK')
ax.axvline(x=100, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Void radius R [Mpc/h]', fontsize=12)
ax.set_ylabel('ΔT [μK]', fontsize=12)
ax.set_title('ISW Signal vs Void Size (δ=-0.4, z=0.5)', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Enhancement factor
ax = axes[0, 1]
enh_scan = np.array([isw_void_gcv(R, -0.4, 0.5)[1] for R in R_scan])
ax.plot(R_scan * h, enh_scan, 'purple', linewidth=2.5)
ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, label='Obs/LCDM ~ 2×')
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='No enhancement')
ax.set_xlabel('Void radius R [Mpc/h]', fontsize=12)
ax.set_ylabel('Enhancement (GCV/LCDM)', fontsize=12)
ax.set_title('ISW Enhancement Factor', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 3: Enhancement vs underdensity
ax = axes[0, 2]
delta_scan = np.linspace(-0.95, -0.05, 50)
enh_delta = np.array([isw_void_gcv(R_void_Mpc, d, 0.5)[1] for d in delta_scan])
ax.plot(delta_scan, enh_delta, 'b-', linewidth=2.5)
ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, label='Observed ratio ~ 2×')
ax.set_xlabel('Void underdensity δ_v', fontsize=12)
ax.set_ylabel('Enhancement (GCV/LCDM)', fontsize=12)
ax.set_title('Enhancement vs Void Depth', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 4: Enhancement vs redshift
ax = axes[1, 0]
z_scan = np.linspace(0.1, 1.5, 50)
enh_z = np.array([isw_void_gcv(R_void_Mpc, -0.4, z)[1] for z in z_scan])
ax.plot(z_scan, enh_z, 'b-', linewidth=2.5)
ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, label='Observed ratio ~ 2×')
ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='z=0.5 (Granett+08)')
ax.set_xlabel('Void redshift z', fontsize=12)
ax.set_ylabel('Enhancement (GCV/LCDM)', fontsize=12)
ax.set_title('Enhancement vs Redshift', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 5: Distribution of enhancements from stacking
ax = axes[1, 1]
ax.hist(enhancements, bins=15, color='blue', alpha=0.7, edgecolor='black')
ax.axvline(x=mean_enhancement, color='red', linestyle='--', linewidth=2,
           label=f'Mean: {mean_enhancement:.2f}×')
ax.axvline(x=2.0, color='green', linestyle='--', linewidth=2,
           label='Observed: ~2×')
ax.set_xlabel('Enhancement factor', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'Enhancement Distribution ({N_voids} voids)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 6: Summary
ax = axes[1, 2]
summary = f"""ISW ANOMALY TEST RESULTS

OBSERVATION (Granett+2008):
  50 supervoids, z~0.5, R~100 Mpc/h
  ΔT = -11.3 ± 3.1 μK (stacked)

LCDM PREDICTION:
  ΔT = {delta_T_lcdm:.2f} μK
  Tension: {residual_lcdm:.1f}σ

GCV UNIFIED PREDICTION:
  ΔT = {delta_T_gcv:.2f} μK
  Enhancement: {enhancement:.2f}×
  Tension: {residual_gcv:.1f}σ

STACKED (50 random voids):
  LCDM: {stacked_lcdm:.2f} μK
  GCV:  {stacked_gcv:.2f} μK

VERDICT:
  GCV predicts enhanced ISW signal
  from voids due to χᵥ < 1,
  in the RIGHT DIRECTION to explain
  the Granett+2008 anomaly.
"""
ax.text(0.05, 0.95, summary, transform=ax.transAxes,
        fontsize=9, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.axis('off')

plt.tight_layout()
plt.savefig('/home/manuel/CascadeProjects/gcv-theory/gcv_gpu_tests/theory/131_ISW_Anomaly_Quantitative.png',
            dpi=150, bbox_inches='tight')
print("Figure saved: 131_ISW_Anomaly_Quantitative.png")
plt.close()

print("\n" + "=" * 75)
print("SCRIPT 131 COMPLETED")
print("=" * 75)
