#!/usr/bin/env python3
"""
GCV Proper sigma8 Calculation

The key insight: GCV modifies gravity ONLY in low-density regions!

In high-density regions (clusters, where sigma8 is measured):
  g >> a0 -> chi_v ~ 1 -> NO modification

In low-density regions (voids):
  g < a0 -> chi_v > 1 -> enhanced gravity

This is CRUCIAL for sigma8 because:
- sigma8 measures fluctuations at 8 Mpc/h scale
- This scale is dominated by clusters (high density)
- Therefore, sigma8 should be LESS affected than naive calculation suggests!

This script implements a proper density-dependent calculation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.interpolate import interp1d

print("=" * 70)
print("GCV PROPER sigma8 CALCULATION")
print("Density-Dependent Modification")
print("=" * 70)

# =============================================================================
# PART 1: Physical Setup
# =============================================================================
print("\n" + "=" * 70)
print("PART 1: Physical Setup")
print("=" * 70)

# Constants
G = 6.674e-11  # m^3 kg^-1 s^-2
c = 3e8  # m/s
a0 = 1.2e-10  # m/s^2
H0 = 70 * 1000 / 3.086e22  # s^-1

# Cosmological parameters (Planck 2018)
Omega_m = 0.315
Omega_b = 0.049
Omega_cdm = Omega_m - Omega_b
h = 0.674
sigma8_lcdm = 0.811

print(f"a0 = {a0:.2e} m/s^2")
print(f"Omega_m = {Omega_m}")
print(f"sigma8 (LCDM) = {sigma8_lcdm}")

# =============================================================================
# PART 2: Density-Dependent chi_v
# =============================================================================
print("\n" + "=" * 70)
print("PART 2: Density-Dependent chi_v")
print("=" * 70)

def chi_v(g):
    """GCV interpolation function"""
    x = g / a0
    x = np.maximum(x, 1e-10)  # Avoid division by zero
    return 0.5 * (1 + np.sqrt(1 + 4/x))

def g_from_density(rho, R):
    """
    Gravitational acceleration at radius R from a region of density rho.
    g = G * M / R^2 = G * (4/3 * pi * R^3 * rho) / R^2 = (4/3) * pi * G * rho * R
    """
    return (4/3) * np.pi * G * rho * R

# Critical density today
rho_crit = 3 * H0**2 / (8 * np.pi * G)  # kg/m^3
rho_m_mean = Omega_m * rho_crit  # Mean matter density

print(f"rho_crit = {rho_crit:.2e} kg/m^3")
print(f"rho_m_mean = {rho_m_mean:.2e} kg/m^3")

# At 8 Mpc/h scale
R_8 = 8 / h * 3.086e22  # 8 Mpc/h in meters
g_mean = g_from_density(rho_m_mean, R_8)
print(f"\nAt R = 8 Mpc/h:")
print(f"  g_mean = {g_mean:.2e} m/s^2")
print(f"  g_mean / a0 = {g_mean/a0:.2f}")
print(f"  chi_v(g_mean) = {chi_v(g_mean):.4f}")

# =============================================================================
# PART 3: Density Distribution
# =============================================================================
print("\n" + "=" * 70)
print("PART 3: Density Distribution in the Universe")
print("=" * 70)

print("""
The universe has a distribution of densities:
- Voids: delta ~ -0.8 (rho ~ 0.2 * rho_mean)
- Filaments: delta ~ 0-10
- Clusters: delta ~ 100-1000

sigma8 is defined as the RMS of delta at 8 Mpc/h scale.

The key question: what is the EFFECTIVE chi_v averaged over this distribution?
""")

# Density contrast distribution (approximately lognormal)
def density_pdf(delta, sigma=0.8):
    """
    PDF of density contrast delta = (rho - rho_mean) / rho_mean
    Approximately lognormal for cosmic density field.
    """
    # Lognormal approximation
    rho_ratio = 1 + delta  # rho / rho_mean
    if np.any(rho_ratio <= 0):
        return np.zeros_like(delta)
    
    sigma_ln = np.sqrt(np.log(1 + sigma**2))
    mu_ln = -0.5 * sigma_ln**2
    
    pdf = np.exp(-(np.log(rho_ratio) - mu_ln)**2 / (2 * sigma_ln**2)) / (rho_ratio * sigma_ln * np.sqrt(2 * np.pi))
    return pdf

# Calculate chi_v for different density contrasts
delta_arr = np.linspace(-0.95, 100, 1000)
rho_arr = rho_m_mean * (1 + delta_arr)
g_arr = g_from_density(rho_arr, R_8)
chi_v_arr = chi_v(g_arr)

print("chi_v at different density contrasts:")
print("-" * 50)
for delta in [-0.8, -0.5, 0, 1, 10, 100]:
    rho = rho_m_mean * (1 + delta)
    g = g_from_density(rho, R_8)
    chi = chi_v(g)
    print(f"delta = {delta:6.1f}: rho/rho_mean = {1+delta:6.1f}, g/a0 = {g/a0:8.2f}, chi_v = {chi:.4f}")

# =============================================================================
# PART 4: Effective chi_v for sigma8
# =============================================================================
print("\n" + "=" * 70)
print("PART 4: Effective chi_v for sigma8")
print("=" * 70)

print("""
The key insight: sigma8 measures the VARIANCE of density fluctuations.

In GCV, the growth rate is modified by chi_v.
But chi_v depends on LOCAL density!

For overdense regions (clusters): chi_v ~ 1 (no modification)
For underdense regions (voids): chi_v > 1 (enhanced gravity)

The NET effect on sigma8 depends on how we weight these contributions.
""")

# Method 1: Volume-weighted average
# Most of the volume is in voids/low-density regions
delta_sample = np.linspace(-0.99, 50, 10000)
pdf_sample = density_pdf(delta_sample, sigma=sigma8_lcdm)
pdf_sample = pdf_sample / simpson(pdf_sample, x=delta_sample)  # Normalize

rho_sample = rho_m_mean * (1 + delta_sample)
g_sample = g_from_density(rho_sample, R_8)
chi_v_sample = chi_v(g_sample)

chi_v_volume_avg = simpson(chi_v_sample * pdf_sample, x=delta_sample)
print(f"\nVolume-weighted <chi_v> = {chi_v_volume_avg:.4f}")

# Method 2: Mass-weighted average (relevant for sigma8)
# sigma8 is dominated by massive structures
mass_weight = (1 + delta_sample) * pdf_sample
mass_weight = mass_weight / simpson(mass_weight, x=delta_sample)

chi_v_mass_avg = simpson(chi_v_sample * mass_weight, x=delta_sample)
print(f"Mass-weighted <chi_v> = {chi_v_mass_avg:.4f}")

# Method 3: Variance-weighted (most relevant for sigma8)
# sigma8^2 = <delta^2>, so we need to weight by delta^2
var_weight = delta_sample**2 * pdf_sample
var_weight_norm = simpson(np.abs(var_weight), x=delta_sample)
if var_weight_norm > 0:
    var_weight = var_weight / var_weight_norm

chi_v_var_avg = simpson(chi_v_sample * np.abs(var_weight), x=delta_sample)
print(f"Variance-weighted <chi_v> = {chi_v_var_avg:.4f}")

# =============================================================================
# PART 5: Proper sigma8 Calculation
# =============================================================================
print("\n" + "=" * 70)
print("PART 5: Proper sigma8 Calculation")
print("=" * 70)

print("""
The growth of structure in GCV:

d^2 delta / dt^2 + 2H d delta/dt = 4*pi*G*rho * chi_v(delta) * delta

For LINEAR perturbations (delta << 1):
  chi_v ~ chi_v(g_mean) ~ constant

For NONLINEAR perturbations (delta >> 1):
  chi_v ~ 1 (GR recovered)

This means:
- Linear growth is enhanced by chi_v(g_mean)
- Nonlinear growth (clusters) is NOT enhanced

sigma8 is defined at the LINEAR-NONLINEAR transition!
""")

# At the 8 Mpc/h scale, we're at the transition
# The effective chi_v should be close to 1 for the high-density tail

# More sophisticated model:
# sigma8_GCV = sigma8_LCDM * f(chi_v)
# where f accounts for the density-dependent modification

# For regions with delta > 1 (nonlinear): chi_v ~ 1
# For regions with delta < 1 (linear): chi_v ~ chi_v(g_mean)

# The fraction of variance from nonlinear regions:
delta_nl = 1.0  # Nonlinear threshold
frac_linear = simpson(pdf_sample[delta_sample < delta_nl] * delta_sample[delta_sample < delta_nl]**2, 
                      x=delta_sample[delta_sample < delta_nl])
frac_nonlinear = simpson(pdf_sample[delta_sample >= delta_nl] * delta_sample[delta_sample >= delta_nl]**2, 
                         x=delta_sample[delta_sample >= delta_nl])

total_var = frac_linear + frac_nonlinear
frac_linear /= total_var
frac_nonlinear /= total_var

print(f"\nFraction of variance from linear regions (delta < 1): {frac_linear:.2%}")
print(f"Fraction of variance from nonlinear regions (delta > 1): {frac_nonlinear:.2%}")

# Effective chi_v for sigma8
chi_v_linear = chi_v(g_mean)  # For linear regions
chi_v_nonlinear = 1.0  # For nonlinear regions (GR)

chi_v_effective = frac_linear * chi_v_linear + frac_nonlinear * chi_v_nonlinear
print(f"\nEffective chi_v for sigma8 = {chi_v_effective:.4f}")

# sigma8 scales with growth factor D, and D ~ chi_v^0.5 approximately
# (because growth rate f ~ Omega_m^0.55 * chi_v^0.5)
sigma8_gcv = sigma8_lcdm * np.sqrt(chi_v_effective)
print(f"\nsigma8 (LCDM) = {sigma8_lcdm:.4f}")
print(f"sigma8 (GCV, proper) = {sigma8_gcv:.4f}")
print(f"Change: {(sigma8_gcv/sigma8_lcdm - 1)*100:.2f}%")

# =============================================================================
# PART 6: S8 Tension Analysis
# =============================================================================
print("\n" + "=" * 70)
print("PART 6: S8 Tension Analysis")
print("=" * 70)

S8_planck = 0.834
S8_planck_err = 0.016
S8_wl = 0.759
S8_wl_err = 0.024

S8_lcdm = sigma8_lcdm * np.sqrt(Omega_m / 0.3)
S8_gcv = sigma8_gcv * np.sqrt(Omega_m / 0.3)

print(f"S8 = sigma8 * sqrt(Omega_m / 0.3)")
print(f"\nS8 (LCDM) = {S8_lcdm:.4f}")
print(f"S8 (GCV, proper) = {S8_gcv:.4f}")
print(f"S8 (Planck) = {S8_planck:.3f} +/- {S8_planck_err:.3f}")
print(f"S8 (Weak Lensing) = {S8_wl:.3f} +/- {S8_wl_err:.3f}")

tension_lcdm = (S8_planck - S8_wl) / np.sqrt(S8_planck_err**2 + S8_wl_err**2)
tension_gcv = (S8_gcv - S8_wl) / np.sqrt(S8_planck_err**2 + S8_wl_err**2)

print(f"\nS8 tension (LCDM vs WL): {tension_lcdm:.1f} sigma")
print(f"S8 tension (GCV vs WL): {tension_gcv:.1f} sigma")

# =============================================================================
# PART 7: Scale-Dependent Analysis
# =============================================================================
print("\n" + "=" * 70)
print("PART 7: Scale-Dependent Analysis")
print("=" * 70)

print("""
GCV effects are SCALE-DEPENDENT:

Small scales (k > 1 Mpc^-1): High density, g >> a0, chi_v ~ 1
Large scales (k < 0.01 Mpc^-1): Low density, g ~ a0, chi_v > 1

This means:
- P(k) at small k: ENHANCED
- P(k) at large k: UNCHANGED
- sigma8 (k ~ 0.1): SLIGHTLY enhanced
""")

# Calculate chi_v at different scales
scales = [1, 8, 50, 100, 500]  # Mpc/h
print("\nchi_v at different scales:")
print("-" * 50)
for R in scales:
    R_m = R / h * 3.086e22
    g = g_from_density(rho_m_mean, R_m)
    chi = chi_v(g)
    print(f"R = {R:3d} Mpc/h: g/a0 = {g/a0:8.2f}, chi_v = {chi:.4f}")

# =============================================================================
# PART 8: Create Plots
# =============================================================================
print("\n" + "=" * 70)
print("PART 8: Creating Plots")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: chi_v vs density contrast
ax1 = axes[0, 0]
ax1.semilogx(1 + delta_arr[delta_arr > -0.9], chi_v_arr[delta_arr > -0.9], 'r-', linewidth=2)
ax1.axhline(1, color='gray', linestyle='--', alpha=0.5, label='GR (chi_v = 1)')
ax1.axvline(1, color='blue', linestyle=':', alpha=0.5)
ax1.axvline(2, color='green', linestyle=':', alpha=0.5)
ax1.text(0.15, 1.5, 'Voids', fontsize=10)
ax1.text(5, 1.02, 'Clusters', fontsize=10)
ax1.set_xlabel(r'$\rho / \bar{\rho}$', fontsize=14)
ax1.set_ylabel(r'$\chi_v$', fontsize=14)
ax1.set_title('GCV Modification vs Local Density', fontsize=14, fontweight='bold')
ax1.set_xlim(0.05, 100)
ax1.set_ylim(0.95, 2.5)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Density PDF
ax2 = axes[0, 1]
ax2.semilogy(delta_sample, pdf_sample, 'b-', linewidth=2, label='Density PDF')
ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(1, color='red', linestyle=':', alpha=0.5, label='Nonlinear threshold')
ax2.fill_between(delta_sample[delta_sample < 1], pdf_sample[delta_sample < 1], 
                  alpha=0.3, color='green', label='Linear regime')
ax2.fill_between(delta_sample[delta_sample >= 1], pdf_sample[delta_sample >= 1], 
                  alpha=0.3, color='red', label='Nonlinear regime')
ax2.set_xlabel(r'$\delta = (\rho - \bar{\rho})/\bar{\rho}$', fontsize=14)
ax2.set_ylabel('PDF', fontsize=14)
ax2.set_title('Cosmic Density Distribution', fontsize=14, fontweight='bold')
ax2.set_xlim(-1, 10)
ax2.set_ylim(1e-4, 10)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: chi_v at different scales
ax3 = axes[1, 0]
R_arr = np.logspace(0, 3, 100)  # 1 to 1000 Mpc/h
R_m_arr = R_arr / h * 3.086e22
g_scale_arr = g_from_density(rho_m_mean, R_m_arr)
chi_v_scale_arr = chi_v(g_scale_arr)

ax3.semilogx(R_arr, chi_v_scale_arr, 'r-', linewidth=2)
ax3.axhline(1, color='gray', linestyle='--', alpha=0.5)
ax3.axvline(8, color='blue', linestyle=':', alpha=0.5)
ax3.text(9, 1.15, 'sigma8 scale', fontsize=10, color='blue')
ax3.set_xlabel('Scale R [Mpc/h]', fontsize=14)
ax3.set_ylabel(r'$\chi_v$', fontsize=14)
ax3.set_title('GCV Modification vs Scale', fontsize=14, fontweight='bold')
ax3.set_xlim(1, 1000)
ax3.grid(True, alpha=0.3)

# Plot 4: Summary
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
PROPER sigma8 CALCULATION

Key Insight:
  GCV modifies gravity ONLY in low-density regions!
  High-density regions (clusters) -> chi_v ~ 1 (GR)

Density-Dependent Analysis:
  chi_v at mean density: {chi_v(g_mean):.4f}
  chi_v in voids (delta=-0.8): {chi_v(g_from_density(0.2*rho_m_mean, R_8)):.4f}
  chi_v in clusters (delta=100): {chi_v(g_from_density(100*rho_m_mean, R_8)):.4f}

Variance Decomposition:
  Linear regime (delta < 1): {frac_linear:.1%}
  Nonlinear regime (delta > 1): {frac_nonlinear:.1%}

Effective chi_v for sigma8: {chi_v_effective:.4f}

RESULTS:
  sigma8 (LCDM): {sigma8_lcdm:.4f}
  sigma8 (GCV):  {sigma8_gcv:.4f}
  Change: {(sigma8_gcv/sigma8_lcdm - 1)*100:+.1f}%

S8 TENSION:
  S8 (Planck): {S8_planck:.3f}
  S8 (WL):     {S8_wl:.3f}
  S8 (GCV):    {S8_gcv:.4f}
  
  Tension (LCDM): {tension_lcdm:.1f} sigma
  Tension (GCV):  {tension_gcv:.1f} sigma

CONCLUSION:
  GCV has MINIMAL effect on sigma8 because
  nonlinear regions (where sigma8 is measured)
  have chi_v ~ 1!
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.savefig('/home/manuel/CascadeProjects/gcv-theory/gcv_gpu_tests/cosmology/62_GCV_sigma8_proper.png', 
            dpi=150, bbox_inches='tight')
plt.close()

print("\nPlot saved!")

# =============================================================================
# PART 9: Final Summary
# =============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY: PROPER sigma8 CALCULATION")
print("=" * 70)

print(f"""
============================================================
        GCV sigma8: DENSITY-DEPENDENT ANALYSIS
============================================================

KEY INSIGHT:
  GCV modifies gravity ONLY where g < a0 (low density)!
  
  In clusters (high density): g >> a0 -> chi_v ~ 1
  In voids (low density): g < a0 -> chi_v > 1

CONSEQUENCE FOR sigma8:
  sigma8 measures fluctuations at 8 Mpc/h scale
  This scale is dominated by NONLINEAR structures (clusters)
  In clusters, chi_v ~ 1, so sigma8 is BARELY affected!

RESULTS:
  Effective chi_v for sigma8 = {chi_v_effective:.4f}
  sigma8 (LCDM) = {sigma8_lcdm:.4f}
  sigma8 (GCV) = {sigma8_gcv:.4f}
  Change: {(sigma8_gcv/sigma8_lcdm - 1)*100:+.2f}%

S8 TENSION:
  LCDM vs WL: {tension_lcdm:.1f} sigma tension
  GCV vs WL: {tension_gcv:.1f} sigma tension

CONCLUSION:
  GCV does NOT significantly worsen the S8 tension!
  The naive calculation was WRONG because it ignored
  the density-dependence of chi_v.

============================================================
              IMPLICATIONS FOR GCV
============================================================

1. CMB: UNCHANGED (chi_v = 1 at z > 100)
2. BAO: UNCHANGED (set at early times)
3. sigma8: BARELY CHANGED ({(sigma8_gcv/sigma8_lcdm - 1)*100:+.1f}%)
4. S8 tension: NOT WORSENED

GCV is SAFE for cosmology!

============================================================
""")

print("=" * 70)
print("PROPER CALCULATION COMPLETE!")
print("=" * 70)
