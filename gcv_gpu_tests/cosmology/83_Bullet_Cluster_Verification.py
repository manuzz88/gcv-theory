#!/usr/bin/env python3
"""
BULLET CLUSTER - VERIFICATION OF CALCULATIONS

Let's carefully verify each step of the calculation.
"""

import numpy as np
from scipy.integrate import quad

print("=" * 70)
print("BULLET CLUSTER - CALCULATION VERIFICATION")
print("=" * 70)

# =============================================================================
# Constants
# =============================================================================

G = 6.674e-11  # m^3/kg/s^2
M_sun = 1.989e30  # kg
kpc = 3.086e19  # m
a0 = 1.2e-10  # m/s^2

# =============================================================================
# Step 1: Verify Observational Data
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: Observational Data (from literature)")
print("=" * 70)

# From Clowe et al. (2006), Markevitch et al. (2004)
print("""
OBSERVED VALUES (Clowe et al. 2006, Bradac et al. 2006):

Main cluster:
  - Total lensing mass (within ~1 Mpc): ~1.5 x 10^15 M_sun
  - Gas mass (X-ray): ~1.2 x 10^14 M_sun  
  - Stellar mass: ~3 x 10^13 M_sun
  - Baryon fraction: ~10%

Bullet subcluster:
  - Total lensing mass: ~1.5 x 10^14 M_sun
  - Gas mass: ~2 x 10^13 M_sun (much stripped)
  - Stellar mass: ~1 x 10^13 M_sun
  - Baryon fraction: ~20%

KEY POINT: The lensing mass is ~10x the baryonic mass!
This is the "missing mass" that CDM explains.
""")

M_baryon_main = 1.5e14 * M_sun  # Gas + stars
M_lens_main = 1.5e15 * M_sun    # From weak lensing

print(f"M_baryon (main) = {M_baryon_main/M_sun:.2e} M_sun")
print(f"M_lens (main) = {M_lens_main/M_sun:.2e} M_sun")
print(f"Ratio M_lens/M_baryon = {M_lens_main/M_baryon_main:.1f}")

# =============================================================================
# Step 2: Simple Point Mass Calculation
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: Simple Point Mass Calculation")
print("=" * 70)

def chi_v(g_over_a0):
    """GCV enhancement factor"""
    x = np.maximum(g_over_a0, 1e-10)
    return 0.5 * (1 + np.sqrt(1 + 4/x))

# At what radius is the lensing mass measured?
R_lens = 1000 * kpc  # ~1 Mpc

# Newtonian acceleration from baryons at R_lens
g_N = G * M_baryon_main / R_lens**2
print(f"\nAt R = {R_lens/kpc:.0f} kpc:")
print(f"  g_N = G * M_baryon / R^2 = {g_N:.2e} m/s^2")
print(f"  g_N / a0 = {g_N/a0:.3f}")

# chi_v at this acceleration
cv = chi_v(g_N / a0)
print(f"  chi_v = {cv:.3f}")

# Effective mass in GCV
M_eff_simple = M_baryon_main * cv
print(f"\nSimple GCV prediction:")
print(f"  M_eff = M_baryon * chi_v = {M_eff_simple/M_sun:.2e} M_sun")
print(f"  M_lens (observed) = {M_lens_main/M_sun:.2e} M_sun")
print(f"  Ratio = {M_eff_simple/M_lens_main:.2f}")

# =============================================================================
# Step 3: What chi_v is Needed?
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: What chi_v is Needed?")
print("=" * 70)

chi_v_needed = M_lens_main / M_baryon_main
print(f"chi_v needed = M_lens / M_baryon = {chi_v_needed:.1f}")

# What g/a0 gives this chi_v?
# chi_v = 0.5 * (1 + sqrt(1 + 4/x))
# 2*chi_v - 1 = sqrt(1 + 4/x)
# (2*chi_v - 1)^2 = 1 + 4/x
# x = 4 / ((2*chi_v - 1)^2 - 1)

x_needed = 4 / ((2*chi_v_needed - 1)**2 - 1)
print(f"This requires g/a0 = {x_needed:.4f}")
print(f"i.e., g = {x_needed * a0:.2e} m/s^2")

# What radius would give this g?
r_for_needed_g = np.sqrt(G * M_baryon_main / (x_needed * a0))
print(f"This g occurs at r = {r_for_needed_g/kpc:.0f} kpc")

# =============================================================================
# Step 4: The Problem
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: The Problem")
print("=" * 70)

print(f"""
THE ISSUE:

At R = {R_lens/kpc:.0f} kpc:
  g/a0 = {g_N/a0:.3f}
  chi_v = {cv:.2f}

But we NEED:
  chi_v = {chi_v_needed:.1f}
  which requires g/a0 = {x_needed:.4f}

The actual g/a0 is {g_N/a0 / x_needed:.0f}x TOO HIGH!

In other words, the cluster is NOT in the deep MOND regime.
At cluster scales, g ~ a0, so chi_v ~ 1.5-2, not ~10.
""")

# =============================================================================
# Step 5: Check the Previous Calculation Error
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: Finding the Error in Previous Calculation")
print("=" * 70)

print("""
In the previous script (82_Bullet_Cluster_Lensing_GCV.py),
we got M_lens(GCV) = 2.75e15 M_sun, which is HIGHER than observed.

Let me check what went wrong...

The issue was in the MASS NORMALIZATION.
We set M_baryon = 1.5e14 M_sun but then calculated
M(<1000 kpc) = 1.2e15 M_sun from the beta-model!

This means the beta-model was normalized incorrectly.
The total mass in the model was ~8x higher than intended.
""")

# Let's redo with correct normalization
print("\nCorrect calculation:")

# Beta model parameters
r_c = 250 * kpc  # Core radius
beta = 2/3

def rho_beta(r, rho_0, r_c, beta=2/3):
    """Beta-model density profile"""
    return rho_0 * (1 + (r/r_c)**2)**(-3*beta/2)

def M_beta_analytic(r, rho_0, r_c, beta=2/3):
    """
    Enclosed mass for beta=2/3 model.
    M(<r) = 4*pi*rho_0*r_c^3 * [r/r_c - arctan(r/r_c)]
    """
    x = r / r_c
    return 4 * np.pi * rho_0 * r_c**3 * (x - np.arctan(x))

# Find rho_0 such that M(<R_max) = M_baryon_main
R_max = 2000 * kpc

def find_rho_0(M_target, r_c, R_max):
    """Find rho_0 to match target mass"""
    # M(<R_max) = 4*pi*rho_0*r_c^3 * [R_max/r_c - arctan(R_max/r_c)]
    x = R_max / r_c
    factor = 4 * np.pi * r_c**3 * (x - np.arctan(x))
    return M_target / factor

rho_0 = find_rho_0(M_baryon_main, r_c, R_max)
print(f"rho_0 = {rho_0:.2e} kg/m^3")

# Verify
M_check = M_beta_analytic(R_max, rho_0, r_c)
print(f"M(<{R_max/kpc:.0f} kpc) = {M_check/M_sun:.2e} M_sun (should be {M_baryon_main/M_sun:.2e})")

M_at_1Mpc = M_beta_analytic(1000*kpc, rho_0, r_c)
print(f"M(<1000 kpc) = {M_at_1Mpc/M_sun:.2e} M_sun")

# =============================================================================
# Step 6: Correct GCV Calculation
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: Correct GCV Calculation")
print("=" * 70)

def g_of_r(r, rho_0, r_c):
    """Gravitational acceleration at radius r"""
    M_enc = M_beta_analytic(r, rho_0, r_c)
    return G * M_enc / r**2

# Calculate chi_v at different radii
print(f"\nchi_v profile (correct normalization):")
print(f"{'r [kpc]':<12} {'M(<r) [M_sun]':<18} {'g [m/s^2]':<15} {'g/a0':<12} {'chi_v':<10}")
print("-" * 70)

r_values = [100, 200, 300, 500, 750, 1000, 1500, 2000]
for r_kpc in r_values:
    r = r_kpc * kpc
    M_enc = M_beta_analytic(r, rho_0, r_c)
    g = g_of_r(r, rho_0, r_c)
    cv = chi_v(g / a0)
    print(f"{r_kpc:<12} {M_enc/M_sun:<18.2e} {g:<15.2e} {g/a0:<12.3f} {cv:<10.3f}")

# =============================================================================
# Step 7: Effective Lensing Mass (Correct)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: Effective Lensing Mass (Correct)")
print("=" * 70)

# For a simple estimate, use the mass-weighted average chi_v
# M_eff ~ M_baryon * <chi_v>

# But more correctly, we should integrate Sigma_eff

def Sigma_eff_simple(R, rho_0, r_c, z_max_factor=10):
    """
    Effective surface density in GCV.
    Sigma_eff(R) = integral rho(r) * chi_v(g(r)/a0) dz
    """
    z_max = z_max_factor * r_c
    
    def integrand(z):
        r = np.sqrt(R**2 + z**2)
        if r < 1 * kpc:  # Avoid singularity
            r = 1 * kpc
        rho = rho_beta(r, rho_0, r_c)
        g = g_of_r(r, rho_0, r_c)
        cv = chi_v(g / a0)
        return rho * cv
    
    result, _ = quad(integrand, -z_max, z_max, limit=100)
    return result

def Sigma_baryon(R, rho_0, r_c, z_max_factor=10):
    """Surface density of baryons (no chi_v)"""
    z_max = z_max_factor * r_c
    
    def integrand(z):
        r = np.sqrt(R**2 + z**2)
        return rho_beta(r, rho_0, r_c)
    
    result, _ = quad(integrand, -z_max, z_max, limit=100)
    return result

# Calculate at a few radii
print(f"\nSurface density comparison:")
print(f"{'R [kpc]':<12} {'Sigma_b [kg/m^2]':<20} {'Sigma_eff [kg/m^2]':<20} {'Ratio (chi_v_eff)':<15}")
print("-" * 70)

for R_kpc in [100, 300, 500, 1000]:
    R = R_kpc * kpc
    Sig_b = Sigma_baryon(R, rho_0, r_c)
    Sig_eff = Sigma_eff_simple(R, rho_0, r_c)
    ratio = Sig_eff / Sig_b if Sig_b > 0 else 0
    print(f"{R_kpc:<12} {Sig_b:<20.2e} {Sig_eff:<20.2e} {ratio:<15.2f}")

# =============================================================================
# Step 8: Total Effective Mass
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: Total Effective Lensing Mass")
print("=" * 70)

def M_lens_eff(R_max, rho_0, r_c, n_points=20):
    """
    Total effective lensing mass within projected radius R_max.
    M_eff = 2*pi * integral_0^R_max Sigma_eff(R) * R dR
    """
    R_array = np.linspace(10*kpc, R_max, n_points)
    Sigma_array = np.array([Sigma_eff_simple(R, rho_0, r_c) for R in R_array])
    
    # Trapezoidal integration
    integrand = 2 * np.pi * R_array * Sigma_array
    M_eff = np.trapz(integrand, R_array)
    return M_eff

def M_baryon_projected(R_max, rho_0, r_c, n_points=20):
    """Total baryonic mass within projected radius R_max."""
    R_array = np.linspace(10*kpc, R_max, n_points)
    Sigma_array = np.array([Sigma_baryon(R, rho_0, r_c) for R in R_array])
    
    integrand = 2 * np.pi * R_array * Sigma_array
    M_b = np.trapz(integrand, R_array)
    return M_b

print("Calculating (this may take a moment)...")

R_test = 1000 * kpc
M_eff = M_lens_eff(R_test, rho_0, r_c)
M_b_proj = M_baryon_projected(R_test, rho_0, r_c)

print(f"\nAt R = {R_test/kpc:.0f} kpc:")
print(f"  M_baryon (projected) = {M_b_proj/M_sun:.2e} M_sun")
print(f"  M_eff (GCV) = {M_eff/M_sun:.2e} M_sun")
print(f"  Effective chi_v = {M_eff/M_b_proj:.2f}")
print(f"  M_lens (observed) = {M_lens_main/M_sun:.2e} M_sun")
print(f"  GCV / Observed = {M_eff/M_lens_main:.2f}")

# =============================================================================
# Step 9: Final Summary
# =============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

# Simple estimate
g_at_R = g_of_r(R_test, rho_0, r_c)
chi_v_at_R = chi_v(g_at_R / a0)
M_eff_simple = M_baryon_main * chi_v_at_R

print(f"""
============================================================
        BULLET CLUSTER - VERIFIED CALCULATION
============================================================

OBSERVATIONAL DATA:
  M_baryon = {M_baryon_main/M_sun:.2e} M_sun
  M_lens (observed) = {M_lens_main/M_sun:.2e} M_sun
  Ratio = {M_lens_main/M_baryon_main:.1f}

GCV CALCULATION (at R = {R_test/kpc:.0f} kpc):
  g = {g_at_R:.2e} m/s^2
  g/a0 = {g_at_R/a0:.3f}
  chi_v = {chi_v_at_R:.2f}

GCV PREDICTIONS:
  Simple: M_eff = M_baryon * chi_v = {M_eff_simple/M_sun:.2e} M_sun
  Integrated: M_eff = {M_eff/M_sun:.2e} M_sun

COMPARISON:
  GCV / Observed = {M_eff/M_lens_main:.2f} (integrated)
  GCV / Observed = {M_eff_simple/M_lens_main:.2f} (simple)

WHAT'S NEEDED:
  chi_v = {chi_v_needed:.1f}
  But we get chi_v = {chi_v_at_R:.2f}

DEFICIT:
  GCV explains {M_eff/M_lens_main*100:.0f}% of observed mass
  Missing: {(1 - M_eff/M_lens_main)*100:.0f}%

============================================================

CONCLUSION:
GCV with standard baryonic mass explains ~{M_eff/M_lens_main*100:.0f}% 
of the Bullet Cluster lensing mass.

The remaining ~{(1 - M_eff/M_lens_main)*100:.0f}% would require either:
1. Additional baryonic mass (unlikely at this level)
2. Neutrinos with m_nu >> 0.12 eV (tension with Planck)
3. Some other form of dark matter
4. Modification to GCV at cluster scales

This is consistent with the known MOND cluster problem.

============================================================
""")

print("=" * 70)
print("VERIFICATION COMPLETE!")
print("=" * 70)
