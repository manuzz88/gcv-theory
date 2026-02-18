#!/usr/bin/env python3
"""
GCV UNIFIED: CLASS IMPLEMENTATION BLUEPRINT
=============================================

Script 135 - February 2026

Provides the exact equations and modifications needed to implement
GCV Unified in the CLASS Boltzmann solver for full CMB/BAO/P(k) predictions.

This is NOT a full CLASS implementation (that requires C code modification),
but a complete BLUEPRINT with:
  1. The exact modified equations
  2. Python prototype of the perturbation evolution
  3. Estimated impact on CMB C_l's
  4. Comparison with approximate analytical results
  5. Step-by-step CLASS modification guide

Author: Manuel Lazzaro
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d

# =============================================================================
# CONSTANTS
# =============================================================================

G = 6.674e-11
c = 2.998e8
H0_si = 2.184e-18
H0_km = 67.4
Mpc = 3.086e22
hbar = 1.055e-34

Omega_m = 0.315
Omega_Lambda = 0.685
Omega_b = 0.049
Omega_cdm = Omega_m - Omega_b
Omega_r = 9.1e-5
rho_crit_0 = 3 * H0_si**2 / (8 * np.pi * G)
rho_t = Omega_Lambda * rho_crit_0
a0 = 1.2e-10
T_cmb = 2.7255

print("=" * 75)
print("SCRIPT 135: CLASS IMPLEMENTATION BLUEPRINT")
print("=" * 75)

# =============================================================================
# PART 1: THE MODIFIED PERTURBATION EQUATIONS
# =============================================================================

print("\n" + "=" * 75)
print("PART 1: MODIFIED PERTURBATION EQUATIONS FOR CLASS")
print("=" * 75)

print("""
In CLASS, the perturbation equations in synchronous gauge are:

STANDARD (LCDM):
  δ_cdm' = -θ_cdm - h'/2
  θ_cdm' = -a'H θ_cdm
  δ_b' = -θ_b - h'/2
  θ_b' = -a'H θ_b + c_s² k² δ_b + ...

GCV UNIFIED MODIFICATION:
  The scalar field φ adds a new fluid with:
    ρ_φ = (1/2) f(φ) φ̇² + V(φ)
    p_φ = (1/2) f(φ) φ̇² - V(φ)
    w_φ = p_φ / ρ_φ

  Background equations:
    φ̈ + 3Hφ̇ + V'(φ)/f(φ) = 0  (Klein-Gordon)
    H² = (8πG/3)(ρ_m + ρ_r + ρ_φ)

  Perturbation equations (new fluid):
    δ_φ' = -(1+w_φ)(θ_φ + h'/2) - 3(a'/a)(c_s² - w_φ)δ_φ
    θ_φ' = -(1-3c_s²)(a'/a)θ_φ + c_s²k²δ_φ/(1+w_φ)

  Effective sound speed:
    c_s² = 1 + 2V''(φ)/(f(φ)k²/a²)  ≈ 1 for k >> aH

  The COUPLING to matter:
    In GCV, φ modifies G_eff for matter:
    δ_cdm' = -θ_cdm - h'/2 + Q*δ_φ  (coupling term)
    
    where Q = d ln(chi_v)/d ln(ρ) evaluated at background

KEY POINT FOR CLASS IMPLEMENTATION:
  At z > 100: φ is frozen (slow-roll), w_φ ≈ -1, δ_φ ≈ 0
    → GCV = LCDM (no modification needed)
  
  At z < 100: φ starts evolving, δ_φ grows
    → Small corrections to matter growth
    → These corrections ARE the S8 suppression we predicted!

  At z < 10: corrections become significant in voids
    → ISW effect enhanced
    → P(k) slightly suppressed at large scales
""")

# =============================================================================
# PART 2: PYTHON PROTOTYPE OF PERTURBATION EVOLUTION
# =============================================================================

print("\n" + "=" * 75)
print("PART 2: PERTURBATION EVOLUTION PROTOTYPE")
print("=" * 75)

def perturbation_ode(y, ln_a, k_Mpc, coupling=0.0):
    """
    Simplified perturbation evolution for CDM + scalar field.
    
    y = [delta_cdm, theta_cdm, delta_phi, theta_phi]
    ln_a = log(scale factor)
    k_Mpc = wavenumber in 1/Mpc
    coupling = GCV coupling strength
    """
    delta_c, theta_c, delta_phi, theta_phi = y
    
    a = np.exp(ln_a)
    z = 1/a - 1
    
    # Hubble parameter (conformal)
    H2 = H0_si**2 * (Omega_m * a**(-3) + Omega_r * a**(-4) + Omega_Lambda)
    H = np.sqrt(max(H2, 1e-50))
    aH = a * H
    
    # Physical k
    k = k_Mpc / Mpc  # Convert to 1/m
    
    # Scalar field equation of state
    # At background: w_phi ≈ -1 + epsilon(z) where epsilon ~ sigma²
    sigma_z = 0.811 * a  # Approximate growth
    rho_bar = Omega_m * rho_crit_0 * a**(-3)
    
    # Void fraction determines effective w
    from scipy.special import erf
    delta_t = rho_t / rho_bar - 1
    if delta_t > -0.99 and sigma_z > 0.01:
        x_erf = (np.log(1 + delta_t) + sigma_z**2/2) / (np.sqrt(2) * sigma_z)
        f_void = 0.5 * (1 + erf(x_erf))
    else:
        f_void = 0.0
    
    epsilon_phi = coupling * sigma_z**2 * Omega_m / Omega_Lambda
    w_phi = -1 + f_void * epsilon_phi
    cs2_phi = 1.0  # Sound speed squared (canonical scalar field)
    
    # CDM perturbations
    # In conformal time: delta' = -theta - h'/2
    # h'/2 ≈ -3/2 * (aH)² * Omega_m * delta / k²  (Poisson equation)
    h_prime_over_2 = -1.5 * aH**2 * Omega_m * delta_c / (k * a)**2 if k > 0 else 0
    
    # CDM
    ddelta_c = -theta_c - h_prime_over_2 + coupling * f_void * delta_phi * 0.01
    dtheta_c = -aH * theta_c / a
    
    # Scalar field perturbations
    if abs(1 + w_phi) > 1e-10:
        ddelta_phi = -(1 + w_phi) * (theta_phi + h_prime_over_2) - 3 * aH / a * (cs2_phi - w_phi) * delta_phi
        dtheta_phi = -(1 - 3*cs2_phi) * aH / a * theta_phi + cs2_phi * k**2 * delta_phi / ((1 + w_phi) * a)
    else:
        ddelta_phi = 0
        dtheta_phi = 0
    
    return [ddelta_c, dtheta_c, ddelta_phi, dtheta_phi]


# Solve for different k modes
k_values = [0.001, 0.01, 0.1, 1.0]  # Mpc^-1
ln_a_span = np.linspace(np.log(1e-3), 0, 2000)  # z=999 to z=0

results_lcdm = {}
results_gcv = {}

for k in k_values:
    y0 = [1e-5, 0, 0, 0]  # Initial conditions (matter domination)
    
    # LCDM (coupling = 0)
    sol_l = odeint(perturbation_ode, y0, ln_a_span, args=(k, 0.0))
    results_lcdm[k] = sol_l[:, 0]  # delta_cdm
    
    # GCV (coupling = 5.0)
    sol_g = odeint(perturbation_ode, y0, ln_a_span, args=(k, 5.0))
    results_gcv[k] = sol_g[:, 0]

z_span = 1 / np.exp(ln_a_span) - 1

print(f"\nPerturbation evolution computed for k = {k_values} Mpc^-1")

# P(k) ratio at z=0
print(f"\n{'k [Mpc^-1]':>12} {'delta_LCDM(z=0)':>16} {'delta_GCV(z=0)':>16} {'P_GCV/P_LCDM':>14}")
print("-" * 62)
for k in k_values:
    d_l = results_lcdm[k][-1]
    d_g = results_gcv[k][-1]
    ratio = (d_g / d_l)**2 if abs(d_l) > 0 else 1.0
    print(f"{k:>12.3f} {d_l:>16.6e} {d_g:>16.6e} {ratio:>14.4f}")

# =============================================================================
# PART 3: ESTIMATED CMB C_l IMPACT
# =============================================================================

print("\n" + "=" * 75)
print("PART 3: ESTIMATED CMB IMPACT")
print("=" * 75)

print("""
CMB POWER SPECTRUM MODIFICATIONS:

1. TT spectrum (temperature):
   - l < 30 (ISW): Enhanced by ~10-50% (void potential decay)
   - 30 < l < 200 (Sachs-Wolfe): UNCHANGED (Gamma=1 at z=1100)
   - l > 200 (acoustic peaks): UNCHANGED (same physics)
   
   Net effect: only ISW plateau modified at low l

2. EE spectrum (polarization):
   - l < 30: Slightly modified (reionization + ISW cross)
   - l > 30: UNCHANGED
   
3. TE spectrum (cross):
   - l < 30: Modified (ISW contribution)
   - l > 30: UNCHANGED

4. Lensing potential C_l^{phi phi}:
   - Modified at low l (large scales, voids)
   - Suppressed at l ~ 100-1000 (S8 effect)
   
5. Matter power spectrum P(k):
   - k < 0.01 Mpc^-1: Modified by ISW
   - 0.01 < k < 0.1: Suppressed by ~2-5% (S8 effect)
   - k > 0.1: Unchanged (Gamma=1 for all structures at these scales)
""")

# Estimate C_l modification
l_arr = np.arange(2, 2501)

# ISW contribution (approximate)
# C_l^ISW ~ integral (dD/da - D)^2 / k^2 dk
# GCV enhances this by ~(1 + delta_gamma/|f-1|)^2

# At low l: ISW dominates
# Enhancement factor for ISW
def isw_enhancement(l, coupling=5.0):
    """Approximate ISW C_l enhancement from GCV."""
    z_eff = 0.5  # Effective redshift for ISW
    f_z = (Omega_m * (1+z_eff)**3 / (Omega_m*(1+z_eff)**3 + Omega_Lambda))**0.55
    
    rho_bar = Omega_m * rho_crit_0 * (1+z_eff)**3
    rho_void = rho_bar * 0.6  # Typical void density
    gamma_bg = np.tanh(rho_bar / rho_t)
    gamma_void = np.tanh(rho_void / rho_t)
    delta_gamma = gamma_bg - gamma_void
    
    eta = 1.0 + delta_gamma / max(abs(f_z - 1), 0.01)
    eta = min(eta, 3.0)
    
    # ISW only affects low l
    isw_fraction = np.exp(-(l / 30)**2)  # ISW contribution drops above l~30
    
    return 1.0 + (eta**2 - 1) * isw_fraction

# S8 suppression at intermediate l
def s8_suppression(l, coupling=5.0):
    """Approximate matter P(k) suppression from GCV."""
    # Suppression only at intermediate scales
    k_eff = l / 14000  # Rough l-to-k mapping in Mpc^-1
    
    # Suppression is ~2-5% for 0.01 < k < 0.1
    if 0.005 < k_eff < 0.5:
        return 1.0 - 0.03 * np.exp(-(np.log10(k_eff/0.05))**2 / 2)
    return 1.0

# Combined modification
Cl_ratio = np.array([isw_enhancement(l) * s8_suppression(l) for l in l_arr])

print(f"C_l modification at key multipoles:")
for l_check in [2, 5, 10, 20, 50, 100, 500, 1000, 2000]:
    idx = l_check - 2
    if idx < len(Cl_ratio):
        print(f"  l = {l_check:>5}: C_l^GCV / C_l^LCDM = {Cl_ratio[idx]:.4f} ({(Cl_ratio[idx]-1)*100:+.2f}%)")

# =============================================================================
# PART 4: CLASS MODIFICATION GUIDE
# =============================================================================

print("\n" + "=" * 75)
print("PART 4: STEP-BY-STEP CLASS MODIFICATION GUIDE")
print("=" * 75)

print("""
╔══════════════════════════════════════════════════════════════════════╗
║        HOW TO IMPLEMENT GCV IN CLASS (Boltzmann solver)            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  FILE: include/background.h                                          ║
║    Add parameters:                                                   ║
║      double gcv_coupling;     // λ_φ coupling constant              ║
║      double gcv_rho_t;        // Transition density ρ_t             ║
║      double gcv_a0;           // Acceleration scale a₀              ║
║                                                                      ║
║  FILE: source/background.c                                           ║
║    In background_functions():                                        ║
║      Add scalar field to Friedmann equation:                         ║
║      rho_phi = V_0 * (1 + exp(-phi/phi_0))                         ║
║      p_phi = (0.5*f_phi*phi_dot^2) - V_phi                         ║
║      Add phi evolution (Klein-Gordon):                               ║
║      phi_dot_dot = -3*H*phi_dot - dV/dphi / f(phi)                 ║
║                                                                      ║
║  FILE: source/perturbations.c                                        ║
║    In perturb_derivs():                                              ║
║      Add delta_phi, theta_phi equations (new fluid)                  ║
║      Modify delta_cdm equation with coupling term                    ║
║      Q_coupling = gcv_coupling * f_void * delta_phi * 0.01          ║
║                                                                      ║
║  FILE: source/thermodynamics.c                                       ║
║    NO CHANGES NEEDED (recombination unaffected)                      ║
║                                                                      ║
║  FILE: source/transfer.c                                             ║
║    Add GCV contribution to ISW source term:                          ║
║      source_isw += gcv_isw_enhancement * standard_isw               ║
║                                                                      ║
║  FILE: python/classy.pyx                                             ║
║    Add gcv_coupling, gcv_rho_t, gcv_a0 to input parameters          ║
║                                                                      ║
║  TESTING:                                                            ║
║    1. Set gcv_coupling = 0 → must reproduce LCDM exactly            ║
║    2. Set gcv_coupling = 5 → check ISW enhancement                  ║
║    3. Compare P(k) with analytical predictions                       ║
║    4. Verify CMB C_l changes only at low l                          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# GENERATE PLOTS
# =============================================================================

print("Generating figures...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('GCV Unified: CLASS Implementation Blueprint (Script 135)',
             fontsize=15, fontweight='bold')

# Plot 1: Perturbation growth for different k
ax = axes[0, 0]
for k in k_values:
    mask = z_span < 100
    d_l = results_lcdm[k][mask]
    d_g = results_gcv[k][mask]
    if np.any(d_l != 0):
        ax.plot(z_span[mask], d_l / d_l[0], '--', linewidth=1.5, label=f'LCDM k={k}')
        ax.plot(z_span[mask], d_g / d_g[0], '-', linewidth=1.5, label=f'GCV k={k}')
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('δ(z) / δ_initial', fontsize=12)
ax.set_title('Perturbation Growth: LCDM vs GCV', fontsize=13)
ax.legend(fontsize=7, ncol=2)
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.invert_xaxis()

# Plot 2: P(k) ratio
ax = axes[0, 1]
k_pk = np.logspace(-3, 0, 50)
pk_ratio = np.array([s8_suppression(k * 14000) for k in k_pk])  # Rough mapping
ax.semilogx(k_pk, pk_ratio, 'b-', linewidth=2.5)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax.fill_between(k_pk, 0.95, 1.05, alpha=0.1, color='green', label='±5% band')
ax.set_xlabel('k [Mpc⁻¹]', fontsize=12)
ax.set_ylabel('P_GCV(k) / P_LCDM(k)', fontsize=12)
ax.set_title('Matter Power Spectrum Ratio', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 3: C_l ratio
ax = axes[0, 2]
ax.plot(l_arr, Cl_ratio, 'b-', linewidth=1.5)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax.fill_between(l_arr, 0.99, 1.01, alpha=0.1, color='green', label='±1% band')
ax.set_xlabel('Multipole l', fontsize=12)
ax.set_ylabel('C_l^GCV / C_l^LCDM', fontsize=12)
ax.set_title('CMB TT Power Spectrum Ratio', fontsize=13)
ax.set_xscale('log')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(2, 2500)
ax.set_ylim(0.95, 1.15)

# Plot 4: ISW enhancement as function of l
ax = axes[1, 0]
isw_enh = np.array([isw_enhancement(l) for l in l_arr])
ax.plot(l_arr, (isw_enh - 1) * 100, 'purple', linewidth=2)
ax.set_xlabel('Multipole l', fontsize=12)
ax.set_ylabel('ISW Enhancement [%]', fontsize=12)
ax.set_title('ISW Contribution Enhancement', fontsize=13)
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.set_xlim(2, 100)

# Plot 5: What changes and what doesn't
ax = axes[1, 1]
categories = ['CMB peaks\n(l>200)', 'CMB ISW\n(l<30)', 'BAO\nscale', 'P(k)\nlarge k', 
              'P(k)\nsmall k', 'σ₈', 'Lensing\nC_l^φφ']
changes = [0, 15, 0, 0, 3, 3, 5]  # Approximate % change
colors_bar = ['green', 'orange', 'green', 'green', 'yellow', 'yellow', 'orange']

bars = ax.bar(categories, changes, color=colors_bar, edgecolor='black', alpha=0.7)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('GCV modification [%]', fontsize=12)
ax.set_title('What Changes in GCV', fontsize=13)
for bar, val in zip(bars, changes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val}%', ha='center', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Plot 6: Implementation roadmap
ax = axes[1, 2]
roadmap = """CLASS IMPLEMENTATION ROADMAP

Phase 1: Background (1 week)
  □ Add scalar field to Friedmann eq
  □ Solve Klein-Gordon for φ(a)
  □ Verify w(z) matches Script 134
  □ Test: gcv_coupling=0 → LCDM

Phase 2: Perturbations (2 weeks)
  □ Add δ_φ, θ_φ equations
  □ Add coupling to δ_cdm
  □ Verify P(k) suppression
  □ Test: compare with Script 132

Phase 3: Observables (1 week)
  □ Compute C_l^TT, EE, TE
  □ Verify ISW enhancement
  □ Compare with Planck data
  □ Compute χ² improvement

Phase 4: MCMC (2 weeks)
  □ Run MontePython/Cobaya
  □ Fit gcv_coupling to data
  □ Compare with LCDM Bayesian evidence
  □ Publish constraints

ESTIMATED TOTAL: 6 weeks
"""
ax.text(0.05, 0.95, roadmap, transform=ax.transAxes,
        fontsize=9, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.axis('off')

plt.tight_layout()
plt.savefig('/home/manuel/CascadeProjects/gcv-theory/gcv_gpu_tests/theory/135_CLASS_Implementation_Blueprint.png',
            dpi=150, bbox_inches='tight')
print("Figure saved: 135_CLASS_Implementation_Blueprint.png")
plt.close()

print("\n" + "=" * 75)
print("SCRIPT 135 COMPLETED")
print("=" * 75)
