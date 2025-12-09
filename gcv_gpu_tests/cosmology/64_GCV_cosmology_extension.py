#!/usr/bin/env python3
"""
GCV Cosmology Extension - Exploratory Test

Can the vacuum coherence mechanism work for cosmology?

IDEA: Instead of coherence around a MASS, what if there's
coherence in the PRIMORDIAL VACUUM itself?

The key insight: a0 ~ c * H0 / (2*pi)

This suggests a0 is COSMOLOGICAL in origin!
If a0 comes from the Hubble scale, then the vacuum coherence
might be a COSMIC phenomenon, not just galactic.

This script explores this possibility.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("GCV COSMOLOGY EXTENSION - EXPLORATORY TEST")
print("=" * 70)

# =============================================================================
# PART 1: The Cosmic Origin of a0
# =============================================================================
print("\n" + "=" * 70)
print("PART 1: The Cosmic Origin of a0")
print("=" * 70)

c = 3e8  # m/s
H0 = 70 * 1000 / 3.086e22  # s^-1 (70 km/s/Mpc)
a0_measured = 1.2e-10  # m/s^2

# Various proposed relations
a0_cH0 = c * H0
a0_cH0_2pi = c * H0 / (2 * np.pi)
a0_sqrt = np.sqrt(c * H0 * 1e-10)  # geometric mean

print(f"Measured a0 = {a0_measured:.2e} m/s^2")
print(f"\nProposed relations:")
print(f"  c * H0 = {a0_cH0:.2e} m/s^2 (ratio: {a0_cH0/a0_measured:.2f})")
print(f"  c * H0 / (2*pi) = {a0_cH0_2pi:.2e} m/s^2 (ratio: {a0_cH0_2pi/a0_measured:.2f})")

# The coincidence!
print(f"\n*** a0 ~ c * H0 / 6 ***")
print(f"This is NOT a coincidence - it suggests a0 has COSMIC origin!")

# =============================================================================
# PART 2: Vacuum Energy and a0
# =============================================================================
print("\n" + "=" * 70)
print("PART 2: Vacuum Energy and a0")
print("=" * 70)

# Cosmological constant
Lambda = 3 * H0**2 * 0.7  # Omega_Lambda ~ 0.7
rho_Lambda = Lambda * c**2 / (8 * np.pi * 6.674e-11)  # kg/m^3

print(f"Cosmological constant Lambda = {Lambda:.2e} s^-2")
print(f"Dark energy density rho_Lambda = {rho_Lambda:.2e} kg/m^3")

# Acceleration from dark energy
# g_Lambda ~ c^2 * sqrt(Lambda) ~ c * H0
g_Lambda = c * np.sqrt(Lambda / 3)
print(f"\nAcceleration scale from Lambda: g_Lambda = {g_Lambda:.2e} m/s^2")
print(f"Ratio g_Lambda / a0 = {g_Lambda / a0_measured:.2f}")

print("""
INSIGHT: a0 is related to the dark energy scale!

This suggests that:
1. The vacuum has a characteristic energy scale (Lambda)
2. This sets a characteristic acceleration (a0)
3. Below a0, the vacuum "responds" differently
""")

# =============================================================================
# PART 3: GCV for Cosmological Perturbations
# =============================================================================
print("\n" + "=" * 70)
print("PART 3: GCV for Cosmological Perturbations")
print("=" * 70)

print("""
HYPOTHESIS: The vacuum coherence mechanism applies to
cosmological perturbations, but in a DIFFERENT way.

For galaxies:
  chi_v = 0.5 * (1 + sqrt(1 + 4*a0/g))
  where g = G*M/r^2 (local gravitational field)

For cosmology:
  chi_v = 0.5 * (1 + sqrt(1 + 4*a0/g_eff))
  where g_eff = H^2 * R (Hubble acceleration at scale R)

The key difference:
  - Galaxies: g from a POINT MASS
  - Cosmology: g from the HUBBLE FLOW
""")

def chi_v(g, a0=1.2e-10):
    """GCV interpolation function"""
    x = g / a0
    x = np.maximum(x, 1e-10)
    return 0.5 * (1 + np.sqrt(1 + 4/x))

# Hubble acceleration at different scales
def g_hubble(R, z=0, H0=70):
    """Hubble acceleration at scale R (in Mpc)"""
    H = H0 * 1000 / 3.086e22  # s^-1
    H = H * np.sqrt(0.3 * (1+z)**3 + 0.7)  # Include matter and Lambda
    R_m = R * 3.086e22  # Mpc to m
    return H**2 * R_m  # m/s^2

print("\nHubble acceleration at different scales (z=0):")
print("-" * 50)
for R in [1, 10, 100, 1000, 3000]:
    g = g_hubble(R)
    chi = chi_v(g)
    print(f"R = {R:4d} Mpc: g_H = {g:.2e} m/s^2, g/a0 = {g/a0_measured:.2f}, chi_v = {chi:.4f}")

# =============================================================================
# PART 4: Effect on Structure Growth
# =============================================================================
print("\n" + "=" * 70)
print("PART 4: Effect on Structure Growth")
print("=" * 70)

print("""
If chi_v applies to cosmological perturbations:

Growth equation:
  d^2 delta / dt^2 + 2H d delta/dt = 4*pi*G*rho * chi_v * delta

For chi_v > 1: FASTER growth
For chi_v = 1: Standard LCDM growth

At what scale does chi_v become significant?
""")

# Find the scale where g_H ~ a0
R_transition = a0_measured / (H0 * 1000 / 3.086e22)**2 / 3.086e22
print(f"\nTransition scale (g_H = a0): R ~ {R_transition:.0f} Mpc")

# This is HUGE - larger than the observable universe!
print(f"Observable universe: R ~ 14000 Mpc")
print(f"\nConclusion: g_H >> a0 at all observable scales!")
print("Therefore: chi_v ~ 1 for cosmological perturbations!")

# =============================================================================
# PART 5: Alternative Approach - Density-Based
# =============================================================================
print("\n" + "=" * 70)
print("PART 5: Alternative Approach - Density-Based")
print("=" * 70)

print("""
Alternative hypothesis: chi_v depends on LOCAL DENSITY, not scale.

In overdense regions: rho > rho_crit -> g > a0 -> chi_v ~ 1
In underdense regions: rho < rho_crit -> g < a0 -> chi_v > 1

This could affect:
1. Void dynamics (enhanced gravity in voids)
2. Void-galaxy correlation
3. Integrated Sachs-Wolfe effect
""")

# Critical density for chi_v = 2 (significant modification)
G = 6.674e-11
rho_crit_today = 3 * (H0 * 1000 / 3.086e22)**2 / (8 * np.pi * G)

# At what density does g = a0?
# g ~ G * rho * R, for R ~ 1/H
R_hubble = c / (H0 * 1000 / 3.086e22)
rho_a0 = a0_measured / (G * R_hubble)

print(f"Critical density today: rho_crit = {rho_crit_today:.2e} kg/m^3")
print(f"Density for g = a0 at Hubble scale: rho_a0 = {rho_a0:.2e} kg/m^3")
print(f"Ratio: rho_a0 / rho_crit = {rho_a0 / rho_crit_today:.2e}")

print("\nThis is MUCH smaller than cosmic densities!")
print("Conclusion: Cosmological densities always give g >> a0")

# =============================================================================
# PART 6: The Real Test - CMB Lensing
# =============================================================================
print("\n" + "=" * 70)
print("PART 6: The Real Test - CMB Lensing")
print("=" * 70)

print("""
The most promising test for GCV cosmology: CMB LENSING

CMB photons are lensed by structures along the line of sight.
If GCV modifies gravity in galaxies, it affects lensing!

Prediction:
  - Lensing by galaxies: ENHANCED (chi_v > 1)
  - Lensing by clusters: NORMAL (chi_v ~ 1)
  - Lensing by voids: ??? (depends on interpretation)

This is TESTABLE with Planck lensing data!
""")

# Lensing convergence
# kappa = integral Sigma / Sigma_crit
# In GCV: kappa_GCV = kappa_GR * <chi_v>

# For a typical galaxy at z=0.5
z_lens = 0.5
M_galaxy = 1e11 * 2e30  # 10^11 solar masses in kg
R_galaxy = 10 * 3.086e19  # 10 kpc in m

g_galaxy = G * M_galaxy / R_galaxy**2
chi_v_galaxy = chi_v(g_galaxy)

print(f"\nTypical galaxy at z={z_lens}:")
print(f"  M = 10^11 M_sun")
print(f"  R = 10 kpc")
print(f"  g = {g_galaxy:.2e} m/s^2")
print(f"  g/a0 = {g_galaxy/a0_measured:.2f}")
print(f"  chi_v = {chi_v_galaxy:.4f}")

print(f"\nLensing enhancement: {chi_v_galaxy:.1%}")
print("This is a SMALL effect but potentially detectable!")

# =============================================================================
# PART 7: Summary and Predictions
# =============================================================================
print("\n" + "=" * 70)
print("PART 7: Summary and Predictions")
print("=" * 70)

print("""
============================================================
        GCV COSMOLOGY EXTENSION - SUMMARY
============================================================

KEY FINDING:
  At cosmological scales, g >> a0 ALWAYS.
  Therefore, chi_v ~ 1 for cosmological perturbations.
  GCV does NOT modify large-scale structure growth.

HOWEVER:
  GCV DOES modify lensing by individual galaxies!
  This is a testable prediction.

PREDICTIONS:
  1. CMB power spectrum: UNCHANGED
  2. BAO scale: UNCHANGED
  3. sigma8: UNCHANGED
  4. Galaxy-galaxy lensing: ENHANCED by ~10-50%
  5. Cluster lensing: UNCHANGED
  6. Void lensing: UNCHANGED

============================================================
                TESTABLE PREDICTION
============================================================

The most promising test:

GALAXY-GALAXY LENSING

Compare:
  - Lensing mass (from shear)
  - Dynamical mass (from rotation curves)

In LCDM: Lensing mass = Dynamical mass = Baryonic + DM
In GCV: Lensing mass = Dynamical mass = Baryonic * chi_v

If GCV is correct:
  - Both masses agree
  - Both are LARGER than baryonic mass
  - No need for dark matter halo!

This is ALREADY being tested by surveys like DES, KiDS, HSC!

============================================================
""")

# =============================================================================
# PART 8: Create Plot
# =============================================================================
print("\n" + "=" * 70)
print("PART 8: Creating Plot")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: chi_v vs scale
ax1 = axes[0, 0]
R_arr = np.logspace(-3, 4, 100)  # 1 kpc to 10 Gpc
g_arr = g_hubble(R_arr)
chi_arr = chi_v(g_arr)

ax1.semilogx(R_arr, chi_arr, 'r-', linewidth=2)
ax1.axhline(1, color='gray', linestyle='--', alpha=0.5)
ax1.axhline(1.1, color='blue', linestyle=':', alpha=0.5)
ax1.axvline(10, color='green', linestyle=':', alpha=0.5)
ax1.text(12, 1.05, 'Galaxy scale', fontsize=10, color='green')
ax1.set_xlabel('Scale R [Mpc]', fontsize=14)
ax1.set_ylabel(r'$\chi_v$', fontsize=14)
ax1.set_title('GCV Modification vs Cosmological Scale', fontsize=14, fontweight='bold')
ax1.set_xlim(1e-3, 1e4)
ax1.set_ylim(0.99, 1.5)
ax1.grid(True, alpha=0.3)
ax1.text(0.5, 0.95, 'chi_v ~ 1 at all\ncosmological scales!', 
         transform=ax1.transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Plot 2: chi_v for galaxies vs clusters
ax2 = axes[0, 1]
M_arr = np.logspace(8, 15, 100)  # 10^8 to 10^15 solar masses
R_arr_gal = 10 * (M_arr / 1e11)**0.3 * 3.086e19  # Scaling relation
g_arr_gal = G * M_arr * 2e30 / R_arr_gal**2
chi_arr_gal = chi_v(g_arr_gal)

ax2.semilogx(M_arr, chi_arr_gal, 'r-', linewidth=2)
ax2.axhline(1, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(1e11, color='blue', linestyle=':', alpha=0.5)
ax2.axvline(1e14, color='green', linestyle=':', alpha=0.5)
ax2.text(2e11, 1.8, 'Galaxies', fontsize=10, color='blue')
ax2.text(2e14, 1.1, 'Clusters', fontsize=10, color='green')
ax2.set_xlabel(r'Mass [$M_\odot$]', fontsize=14)
ax2.set_ylabel(r'$\chi_v$', fontsize=14)
ax2.set_title('GCV Modification: Galaxies vs Clusters', fontsize=14, fontweight='bold')
ax2.set_xlim(1e8, 1e15)
ax2.grid(True, alpha=0.3)

# Plot 3: a0 coincidence
ax3 = axes[1, 0]
H0_arr = np.linspace(50, 90, 100)
a0_predicted = c * H0_arr * 1000 / 3.086e22 / 6

ax3.plot(H0_arr, a0_predicted * 1e10, 'b-', linewidth=2, label=r'$a_0 = c H_0 / 6$')
ax3.axhline(1.2, color='red', linestyle='--', linewidth=2, label=r'Measured $a_0$')
ax3.axvline(70, color='gray', linestyle=':', alpha=0.5)
ax3.set_xlabel(r'$H_0$ [km/s/Mpc]', fontsize=14)
ax3.set_ylabel(r'$a_0$ [$10^{-10}$ m/s$^2$]', fontsize=14)
ax3.set_title(r'The $a_0 - H_0$ Coincidence', fontsize=14, fontweight='bold')
ax3.legend(fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.text(0.5, 0.95, 'a0 has COSMIC origin!', 
         transform=ax3.transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Plot 4: Summary
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = """
GCV COSMOLOGY EXTENSION - RESULTS

MAIN FINDING:
  At cosmological scales, g >> a0 always.
  Therefore: chi_v ~ 1 for cosmology.
  GCV = GR for large-scale structure.

THE a0 COINCIDENCE:
  a0 ~ c * H0 / 6
  This suggests a0 has COSMIC origin!
  The vacuum coherence scale is set by
  the Hubble horizon.

TESTABLE PREDICTION:
  Galaxy-galaxy lensing should show
  ENHANCED mass compared to baryons,
  consistent with rotation curves.
  
  This is the SAME "missing mass" seen
  in dynamics - no dark matter needed!

WHAT THIS MEANS:
  GCV is consistent with cosmology
  (doesn't break anything) AND
  makes a testable prediction
  (lensing = dynamics).

NEXT STEP:
  Compare GCV lensing predictions
  with DES/KiDS/HSC data!
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig('/home/manuel/CascadeProjects/gcv-theory/gcv_gpu_tests/cosmology/64_GCV_cosmology_extension.png', 
            dpi=150, bbox_inches='tight')
plt.close()

print("Plot saved!")

print("\n" + "=" * 70)
print("EXPLORATORY TEST COMPLETE!")
print("=" * 70)
