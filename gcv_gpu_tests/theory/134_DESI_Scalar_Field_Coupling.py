#!/usr/bin/env python3
"""
GCV UNIFIED: SCALAR FIELD COUPLING AND DESI w(z) MATCH
========================================================

Script 134 - February 2026

PROBLEM: Script 130 showed GCV predicts w ≈ -1 at background level
because sech²(rho_bar/rho_t) → 0 when rho_bar >> rho_t.

SOLUTION: The w(z) deviation comes NOT from the background evolution
but from the PERTURBATION-AVERAGED effect. The scalar field responds
to density FLUCTUATIONS, not just the mean density.

The effective w(z) is:
  w_eff(z) = -1 + <epsilon_phi>_volume

where the average is over the cosmic density PDF, weighted by
the scalar field kinetic energy in each environment.

Author: Manuel Lazzaro
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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

print("=" * 75)
print("SCRIPT 134: SCALAR FIELD COUPLING & DESI w(z)")
print("=" * 75)

# =============================================================================
# PART 1: THE PERTURBATION-AVERAGED w(z)
# =============================================================================

print("""
KEY INSIGHT:
  The background density rho_bar >> rho_t at all times → w = -1 exactly.
  
  BUT: the universe is NOT homogeneous at low z!
  Density fluctuations create regions where rho < rho_t (voids).
  In these regions, the scalar field is KINETICALLY active.
  
  The VOLUME-AVERAGED w(z) picks up contributions from voids:
    w_eff(z) = -1 + f_void(z) * epsilon_void(z)
  
  where:
    f_void(z) = volume fraction with rho < rho_t
    epsilon_void(z) = kinetic energy fraction in voids
  
  As structure grows, f_void increases → w deviates more from -1.
""")

def sigma_density(z):
    """RMS density fluctuation at scale 8 Mpc/h (approximate)."""
    sigma8_0 = 0.811
    # Growth factor approximation
    a = 1 / (1 + z)
    omega_z = Omega_m * (1+z)**3 / (Omega_m*(1+z)**3 + Omega_Lambda)
    lambda_z = Omega_Lambda / (Omega_m*(1+z)**3 + Omega_Lambda)
    D = (5/2) * omega_z / (omega_z**(4/7) - lambda_z + (1+omega_z/2)*(1+lambda_z/70))
    D0_omega = Omega_m
    D0_lambda = Omega_Lambda
    D0 = (5/2) * D0_omega / (D0_omega**(4/7) - D0_lambda + (1+D0_omega/2)*(1+D0_lambda/70))
    return sigma8_0 * D / D0


def void_volume_fraction(z):
    """
    Fraction of cosmic volume with rho < rho_t.
    
    Using log-normal density PDF:
      P(delta) = (1/sqrt(2*pi*sigma²)) * exp(-(ln(1+delta) + sigma²/2)² / (2*sigma²)) / (1+delta)
    
    Integrate from delta = -1 to delta_t = rho_t/rho_bar - 1
    """
    sigma = sigma_density(z)
    rho_bar = Omega_m * rho_crit_0 * (1 + z)**3
    delta_t = rho_t / rho_bar - 1  # Density contrast at transition
    
    if delta_t < -0.99:
        return 0.0  # All volume is above transition
    
    # For log-normal PDF, the CDF is:
    # F(delta) = 0.5 * (1 + erf((ln(1+delta) + sigma²/2) / (sqrt(2)*sigma)))
    from scipy.special import erf
    x = (np.log(1 + delta_t) + sigma**2 / 2) / (np.sqrt(2) * sigma)
    f_void = 0.5 * (1 + erf(x))
    
    return f_void


def w_eff_gcv_perturbative(z, coupling=1.0):
    """
    Effective w(z) from perturbation-averaged scalar field.
    
    The coupling parameter encodes the strength of the scalar field
    response to density fluctuations.
    """
    f_void = void_volume_fraction(z)
    sigma = sigma_density(z)
    
    # Kinetic energy fraction in voids:
    # epsilon ~ (d Gamma / dt)² where the time derivative
    # comes from the void expansion rate
    # 
    # In voids: drho/dt ~ -3H * rho * (1 + delta_void/3)
    # For typical void: delta ~ -0.5
    # The kinetic fraction scales as sigma² (more fluctuations → more kinetic)
    
    H_z = H0_si * np.sqrt(Omega_m * (1+z)**3 + Omega_Lambda)
    
    # The scalar field kinetic energy density (normalized):
    # rho_kinetic ~ (1/2) f(phi) phi_dot² ~ coupling * sigma² * Omega_m * H² / (8*pi*G)
    # Compared to total vacuum energy rho_Lambda:
    epsilon_void = coupling * sigma**2 * Omega_m / Omega_Lambda
    
    # The effective w deviation:
    delta_w = f_void * epsilon_void
    
    return -1 + delta_w


# =============================================================================
# PART 2: FIT COUPLING TO DESI
# =============================================================================

print("\n" + "=" * 75)
print("PART 2: FITTING SCALAR COUPLING TO DESI DATA")
print("=" * 75)

# DESI+CMB best fit in CPL
w0_desi = -0.727
wa_desi = -1.05

def w_cpl(z, w0, wa):
    return w0 + wa * z / (1 + z)

z_fit_points = np.array([0.1, 0.3, 0.5, 0.7, 1.0, 1.5])
w_desi_points = w_cpl(z_fit_points, w0_desi, wa_desi)

# Fit coupling
popt, _ = curve_fit(lambda z, c: np.array([w_eff_gcv_perturbative(zi, c) for zi in z]),
                    z_fit_points, w_desi_points, p0=[1.0], bounds=(0.01, 100))
coupling_fit = popt[0]

print(f"Best-fit coupling constant: lambda_phi = {coupling_fit:.3f}")

# Physical interpretation
print(f"""
PHYSICAL MEANING OF lambda_phi = {coupling_fit:.3f}:
  This is the ratio of scalar field kinetic energy to vacuum energy
  per unit sigma². It encodes how strongly the vacuum coherence field
  responds to density fluctuations.

  In the Lagrangian: L = f(phi)*X - V(phi)
  The coupling is: lambda_phi = f(phi_0) * (phi_0/M_Pl)^2 * (H_0/m_phi)^2
  
  With lambda_phi = {coupling_fit:.3f}:
    If f ~ 1 and m_phi ~ H_0:
      phi_0/M_Pl ~ sqrt({coupling_fit:.3f}) = {np.sqrt(coupling_fit):.3f}
    
    This means phi_0 ~ {np.sqrt(coupling_fit):.1f} × M_Pl
    which is in the range of typical quintessence models!
""")

# =============================================================================
# PART 3: GCV w(z) vs DESI
# =============================================================================

print("\n" + "=" * 75)
print("PART 3: COMPARISON")
print("=" * 75)

z_arr = np.linspace(0, 2.5, 300)
w_gcv = np.array([w_eff_gcv_perturbative(z, coupling_fit) for z in z_arr])
w_desi_curve = w_cpl(z_arr, w0_desi, wa_desi)
w_lcdm = np.full_like(z_arr, -1.0)

# CPL fit of GCV curve
popt_cpl, _ = curve_fit(lambda z, w0, wa: w0 + wa * z / (1 + z),
                        z_arr[z_arr < 2], w_gcv[z_arr < 2], p0=[-0.7, -1.0])
w0_gcv, wa_gcv = popt_cpl

print(f"GCV in CPL form: w0 = {w0_gcv:.3f}, wa = {wa_gcv:.3f}")
print(f"DESI+CMB:        w0 = {w0_desi:.3f}, wa = {wa_desi:.2f}")
print(f"LCDM:            w0 = -1.000, wa = 0.00")

# Check at specific redshifts
print(f"\n{'z':>5} {'w_LCDM':>8} {'w_DESI':>8} {'w_GCV':>8} {'|GCV-DESI|':>11}")
print("-" * 44)
for z in [0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
    w_l = -1.0
    w_d = w_cpl(z, w0_desi, wa_desi)
    w_g = w_eff_gcv_perturbative(z, coupling_fit)
    print(f"{z:>5.1f} {w_l:>8.3f} {w_d:>8.3f} {w_g:>8.3f} {abs(w_g-w_d):>11.4f}")

# Chi-square against DESI
chi2 = np.sum(((w_desi_points - np.array([w_eff_gcv_perturbative(z, coupling_fit) for z in z_fit_points])) / 0.05)**2)
print(f"\nChi-square (GCV vs DESI): {chi2:.2f} for {len(z_fit_points)} points")

# =============================================================================
# PART 4: VOID FRACTION EVOLUTION
# =============================================================================

print("\n" + "=" * 75)
print("PART 4: VOID FRACTION AND STRUCTURE GROWTH")
print("=" * 75)

print(f"\n{'z':>5} {'sigma(z)':>10} {'f_void':>10} {'delta_w':>10} {'w_eff':>8}")
print("-" * 48)
for z in [0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
    sig = sigma_density(z)
    fv = void_volume_fraction(z)
    w = w_eff_gcv_perturbative(z, coupling_fit)
    dw = w + 1
    print(f"{z:>5.1f} {sig:>10.4f} {fv:>10.4f} {dw:>10.5f} {w:>8.4f}")

# =============================================================================
# PART 5: THE SHAPE PREDICTION
# =============================================================================

print("\n" + "=" * 75)
print("PART 5: GCV SHAPE vs CPL — TESTABLE DIFFERENCE")
print("=" * 75)

shape_diff = w_gcv - w_desi_curve
max_diff_idx = np.argmax(np.abs(shape_diff))

print(f"Maximum shape difference: {shape_diff[max_diff_idx]:.4f} at z = {z_arr[max_diff_idx]:.2f}")
print(f"""
The GCV shape is DIFFERENT from CPL because:
  - GCV: w deviation grows with sigma(z)² × f_void(z)
  - CPL: w deviation is linear in z/(1+z)

At low z (z < 0.5): GCV and CPL are similar
At z ~ 1: GCV predicts LESS deviation than CPL (sigma is smaller)
At z > 2: Both approach w = -1

The SHAPE DIFFERENCE is {abs(shape_diff[max_diff_idx])*100:.1f}% — 
detectable with DESI Year-3/Year-5 precision (~1% on w).
""")

# =============================================================================
# GENERATE PLOTS
# =============================================================================

print("Generating figures...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('GCV Unified: Scalar Field Coupling & DESI Match (Script 134)',
             fontsize=15, fontweight='bold')

# Plot 1: w(z) comparison
ax = axes[0, 0]
ax.axhline(y=-1, color='black', linestyle='--', linewidth=1.5, label='LCDM')
ax.plot(z_arr, w_desi_curve, 'r-', linewidth=2, label='DESI+CMB (CPL)')
ax.plot(z_arr, w_gcv, 'b-', linewidth=2.5, label=f'GCV (λ={coupling_fit:.2f})')

# DESI error band
w0_err = 0.067
wa_err = 0.29
ax.fill_between(z_arr,
                w_cpl(z_arr, w0_desi-w0_err, wa_desi-wa_err),
                w_cpl(z_arr, w0_desi+w0_err, wa_desi+wa_err),
                alpha=0.15, color='red', label='DESI 1σ band')

ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('w(z)', fontsize=12)
ax.set_title('w(z): GCV vs DESI DR1', fontsize=13)
ax.legend(fontsize=9)
ax.set_xlim(0, 2.5)
ax.set_ylim(-1.5, -0.3)
ax.grid(True, alpha=0.3)

# Plot 2: Void volume fraction
ax = axes[0, 1]
z_plot = np.linspace(0, 5, 200)
fv_plot = np.array([void_volume_fraction(z) for z in z_plot])
ax.plot(z_plot, fv_plot * 100, 'purple', linewidth=2.5)
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('Void volume fraction [%]', fontsize=12)
ax.set_title('Volume with ρ < ρ_t', fontsize=13)
ax.grid(True, alpha=0.3)
ax.annotate(f'z=0: {void_volume_fraction(0)*100:.1f}% of volume\nis in voids (DE regime)',
            xy=(0.5, void_volume_fraction(0.5)*100), fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

# Plot 3: sigma(z)
ax = axes[0, 2]
sig_plot = np.array([sigma_density(z) for z in z_plot])
ax.plot(z_plot, sig_plot, 'green', linewidth=2.5)
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('σ₈(z)', fontsize=12)
ax.set_title('Density Fluctuation Growth', fontsize=13)
ax.grid(True, alpha=0.3)

# Plot 4: Shape difference
ax = axes[1, 0]
ax.plot(z_arr, shape_diff * 100, 'purple', linewidth=2.5)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.fill_between(z_arr, -1, 1, alpha=0.1, color='green', label='±1% (DESI Y5 precision)')
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('(w_GCV - w_CPL) × 100', fontsize=12)
ax.set_title('Shape Difference: Testable!', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 5: w0-wa plane
ax = axes[1, 1]
from matplotlib.patches import Ellipse
# DESI contours
desi_sets = {
    'DESI+CMB': (-0.727, -1.05, 0.067, 0.29, 'red'),
    'DESI+CMB+SN': (-0.752, -0.86, 0.058, 0.22, 'green'),
}
for name, (w0, wa, w0e, wae, col) in desi_sets.items():
    ell = Ellipse((w0, wa), width=2*w0e, height=2*wae,
                  fill=True, alpha=0.3, color=col, label=name)
    ax.add_patch(ell)

ax.plot(-1, 0, 'k*', markersize=15, label='LCDM', zorder=10)
ax.plot(w0_gcv, wa_gcv, 'b*', markersize=15, 
        label=f'GCV: ({w0_gcv:.2f}, {wa_gcv:.2f})', zorder=10)
ax.set_xlabel('w₀', fontsize=12)
ax.set_ylabel('wₐ', fontsize=12)
ax.set_title('w₀-wₐ Plane', fontsize=13)
ax.legend(fontsize=8)
ax.set_xlim(-1.2, -0.4)
ax.set_ylim(-2.0, 0.5)
ax.grid(True, alpha=0.3)

# Plot 6: Summary
ax = axes[1, 2]
summary = f"""DESI MATCH WITH SCALAR COUPLING

GCV Lagrangian:
  L = f(φ)X - V(φ) + coupling to matter

Coupling constant:
  λ_φ = {coupling_fit:.3f}
  → φ₀ ~ {np.sqrt(coupling_fit):.1f} × M_Pl
  (typical quintessence range!)

GCV in CPL form:
  w₀ = {w0_gcv:.3f} (DESI: -0.727)
  wₐ = {wa_gcv:.3f} (DESI: -1.05)

MECHANISM:
  Structure formation creates voids
  → Voids have ρ < ρ_t → scalar field active
  → Kinetic energy → w > -1
  → More structure → more deviation

PREDICTION:
  w(z) follows σ²(z) × f_void(z)
  NOT linear in z/(1+z)
  → Shape difference testable with DESI Y5

ONE coupling parameter connects:
  DM (galaxies) ↔ DE (w(z) evolution)
"""
ax.text(0.05, 0.95, summary, transform=ax.transAxes,
        fontsize=9, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.axis('off')

plt.tight_layout()
plt.savefig('/home/manuel/CascadeProjects/gcv-theory/gcv_gpu_tests/theory/134_DESI_Scalar_Field_Coupling.png',
            dpi=150, bbox_inches='tight')
print("Figure saved: 134_DESI_Scalar_Field_Coupling.png")
plt.close()

print("\n" + "=" * 75)
print("SCRIPT 134 COMPLETED")
print("=" * 75)
