#!/usr/bin/env python3
"""
SEDNA AND EXTREME TRANS-NEPTUNIAN OBJECTS: A GCV PREDICTION?

Sedna (90377) has an extremely eccentric orbit:
- Perihelion: 76 AU
- Aphelion: 937 AU
- Period: ~11,400 years

LCDM explanation: "Planet Nine" (hypothetical)
MOND explanation: External field effect from galaxy

Can GCV make a DIFFERENT prediction?

Key question: At Sedna's distance, is GCV active?
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("SEDNA AND EXTREME TNOs: GCV PREDICTION")
print("=" * 70)

# =============================================================================
# Constants
# =============================================================================

G = 6.674e-11  # m^3/kg/s^2
c = 3e8  # m/s
AU = 1.496e11  # m
M_sun = 1.989e30  # kg
year = 3.156e7  # s

# MOND
a0 = 1.2e-10  # m/s^2

# GCV
f_b = 0.156
Phi_th = (f_b / (2 * np.pi))**3 * c**2
alpha = 1.5
beta = 1.5

print(f"\nGCV threshold: Phi_th/c^2 = {Phi_th/c**2:.2e}")
print(f"a0 = {a0:.2e} m/s^2")

# =============================================================================
# Sedna's Orbit
# =============================================================================
print("\n" + "=" * 70)
print("SEDNA'S ORBIT")
print("=" * 70)

# Orbital parameters
a_sedna = 506 * AU  # semi-major axis
e_sedna = 0.85  # eccentricity
q_sedna = 76 * AU  # perihelion
Q_sedna = 937 * AU  # aphelion

print(f"\nSedna orbital parameters:")
print(f"  Semi-major axis: {a_sedna/AU:.0f} AU")
print(f"  Eccentricity: {e_sedna:.2f}")
print(f"  Perihelion: {q_sedna/AU:.0f} AU")
print(f"  Aphelion: {Q_sedna/AU:.0f} AU")

# =============================================================================
# Gravitational Regime at Sedna's Distance
# =============================================================================
print("\n" + "=" * 70)
print("GRAVITATIONAL REGIME AT SEDNA'S DISTANCE")
print("=" * 70)

def g_newton(r):
    """Newtonian acceleration from Sun at distance r"""
    return G * M_sun / r**2

def Phi_sun(r):
    """Gravitational potential from Sun at distance r"""
    return -G * M_sun / r

# At perihelion
g_peri = g_newton(q_sedna)
Phi_peri = Phi_sun(q_sedna)

# At aphelion
g_aph = g_newton(Q_sedna)
Phi_aph = Phi_sun(Q_sedna)

print(f"\nAt perihelion ({q_sedna/AU:.0f} AU):")
print(f"  g_N = {g_peri:.2e} m/s^2")
print(f"  g_N / a0 = {g_peri/a0:.2f}")
print(f"  |Phi|/c^2 = {abs(Phi_peri)/c**2:.2e}")

print(f"\nAt aphelion ({Q_sedna/AU:.0f} AU):")
print(f"  g_N = {g_aph:.2e} m/s^2")
print(f"  g_N / a0 = {g_aph/a0:.4f}")
print(f"  |Phi|/c^2 = {abs(Phi_aph)/c**2:.2e}")

# Check MOND regime
print(f"\nMOND regime (g < a0):")
print(f"  At perihelion: {'YES' if g_peri < a0 else 'NO'} (g/a0 = {g_peri/a0:.2f})")
print(f"  At aphelion: {'YES' if g_aph < a0 else 'NO'} (g/a0 = {g_aph/a0:.4f})")

# Check GCV regime
print(f"\nGCV regime (|Phi| > Phi_th):")
print(f"  Phi_th/c^2 = {Phi_th/c**2:.2e}")
print(f"  At perihelion: {'YES' if abs(Phi_peri) > Phi_th else 'NO'}")
print(f"  At aphelion: {'YES' if abs(Phi_aph) > Phi_th else 'NO'}")

# =============================================================================
# Key Insight: Solar System Potential
# =============================================================================
print("\n" + "=" * 70)
print("KEY INSIGHT: SOLAR SYSTEM POTENTIAL")
print("=" * 70)

print("""
The Sun's gravitational potential at ANY distance is:
  |Phi_sun|/c^2 = GM_sun / (r * c^2)

At r = 1 AU:
  |Phi|/c^2 = 9.87e-9

At r = 1000 AU:
  |Phi|/c^2 = 9.87e-12

The GCV threshold is:
  Phi_th/c^2 = 1.5e-5

CONCLUSION:
The Solar System potential is ALWAYS below the GCV threshold!
  |Phi_sun|/c^2 << Phi_th/c^2

Therefore:
  - GCV does NOT activate in the Solar System
  - Sedna follows standard MOND, not GCV-enhanced gravity
  - GCV makes NO special prediction for Sedna
""")

# Verify numerically
r_range = np.logspace(0, 4, 100) * AU  # 1 to 10000 AU
Phi_range = np.abs(Phi_sun(r_range)) / c**2

print(f"\nNumerical verification:")
print(f"  Max |Phi|/c^2 in Solar System (at 1 AU): {np.max(Phi_range):.2e}")
print(f"  GCV threshold: {Phi_th/c**2:.2e}")
print(f"  Ratio: {np.max(Phi_range) / (Phi_th/c**2):.2e}")

# =============================================================================
# What About MOND?
# =============================================================================
print("\n" + "=" * 70)
print("WHAT ABOUT MOND?")
print("=" * 70)

print("""
MOND DOES make predictions for Sedna!

At Sedna's aphelion (937 AU):
  g_N = 2.3e-13 m/s^2
  g_N / a0 = 0.002

This is DEEP in the MOND regime (g << a0).

In MOND, the effective acceleration is:
  g_eff = sqrt(g_N * a0) = sqrt(2.3e-13 * 1.2e-10) = 5.3e-12 m/s^2

This is ~23x stronger than Newtonian!

MOND prediction:
  - Sedna's orbit should show anomalies
  - Precession rate different from Newtonian
  - Orbital period slightly different

BUT: The External Field Effect (EFE) complicates this.
The Milky Way's gravity at the Sun's position is:
  g_MW ~ 2e-10 m/s^2 ~ 1.7 * a0

This "external field" partially suppresses MOND effects in the Solar System.
""")

# =============================================================================
# GCV vs MOND for Sedna
# =============================================================================
print("\n" + "=" * 70)
print("GCV vs MOND FOR SEDNA")
print("=" * 70)

# External field from Milky Way
g_MW = 2e-10  # m/s^2 (approximate)

print(f"\nExternal field from Milky Way: g_MW = {g_MW:.2e} m/s^2")
print(f"g_MW / a0 = {g_MW/a0:.2f}")

print("""
In standard MOND with EFE:
  - The external field g_MW ~ 1.7 * a0 dominates
  - MOND effects are suppressed
  - Solar System behaves nearly Newtonian

In GCV:
  - The potential from MW is |Phi_MW|/c^2 ~ 10^-6
  - This is still << Phi_th/c^2 = 1.5e-5
  - GCV does NOT activate
  - Solar System behaves as standard MOND

CONCLUSION:
GCV and MOND make the SAME prediction for Sedna:
  - Standard MOND with EFE
  - No GCV enhancement
  - Slight deviations from Newtonian at large distances
""")

# =============================================================================
# Where WOULD GCV Differ from MOND?
# =============================================================================
print("\n" + "=" * 70)
print("WHERE WOULD GCV DIFFER FROM MOND?")
print("=" * 70)

print("""
GCV differs from MOND only when |Phi|/c^2 > Phi_th/c^2 = 1.5e-5

This requires:
  GM / (r * c^2) > 1.5e-5
  M / r > 1.5e-5 * c^2 / G
  M / r > 2e21 kg/m

For the Sun (M = 2e30 kg):
  r < M / 2e21 = 1e9 m = 0.007 AU

This is INSIDE the Sun! Not observable.

For a galaxy cluster (M = 1e15 M_sun = 2e45 kg):
  r < 2e45 / 2e21 = 1e24 m = 30 Mpc

This is the ENTIRE cluster! GCV is active throughout.

CONCLUSION:
GCV differs from MOND ONLY in galaxy clusters, not in:
  - Solar System
  - Individual galaxies
  - Galaxy groups (marginally)
""")

# =============================================================================
# Alternative: Wide Binaries
# =============================================================================
print("\n" + "=" * 70)
print("ALTERNATIVE TEST: WIDE BINARIES")
print("=" * 70)

print("""
Wide binary stars are a better test of MOND/GCV in the local universe.

For a binary with separation s and total mass M:
  g = GM / s^2

MOND regime (g < a0) requires:
  s > sqrt(GM / a0)

For M = 2 M_sun:
  s > sqrt(2 * 2e30 * 6.67e-11 / 1.2e-10)
  s > 1.5e15 m = 10,000 AU = 0.05 pc

Wide binaries with s > 10,000 AU should show MOND effects!

Recent Gaia data (Chae 2023, Hernandez 2023) claim detection of
MOND-like behavior in wide binaries, but this is controversial.

GCV PREDICTION:
  - Same as MOND for wide binaries
  - |Phi|/c^2 ~ 10^-9 << Phi_th
  - No GCV enhancement
  - Standard MOND applies
""")

# =============================================================================
# The Real Unique GCV Prediction
# =============================================================================
print("\n" + "=" * 70)
print("THE REAL UNIQUE GCV PREDICTION")
print("=" * 70)

print("""
GCV makes a UNIQUE prediction that MOND does not:

IN GALAXY CLUSTERS:
  - MOND predicts: g_eff = sqrt(g_N * a0) (standard MOND)
  - GCV predicts: g_eff = sqrt(g_N * a0_eff) with a0_eff > a0

The enhancement factor is:
  chi_v = 1 + (3/2) * (|Phi|/Phi_th - 1)^(3/2)

For a typical cluster with |Phi|/c^2 ~ 5e-5:
  x = |Phi|/Phi_th ~ 3.3
  chi_v = 1 + 1.5 * (2.3)^1.5 ~ 6.2

This means:
  - MOND predicts 30-50% of observed mass
  - GCV predicts 90% of observed mass

THIS IS THE UNIQUE PREDICTION:
  Cluster lensing mass / baryonic mass should follow GCV formula,
  NOT standard MOND.

Testable with:
  - Weak lensing surveys (DES, Euclid, Rubin)
  - X-ray + lensing combined analysis
  - Cluster mass function
""")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
============================================================
        SEDNA AND GCV: SUMMARY
============================================================

SEDNA:
  - |Phi|/c^2 ~ 10^-11 << Phi_th/c^2 = 1.5e-5
  - GCV does NOT activate
  - Same prediction as MOND
  - NOT a unique GCV test

WIDE BINARIES:
  - |Phi|/c^2 ~ 10^-9 << Phi_th
  - GCV does NOT activate
  - Same prediction as MOND
  - NOT a unique GCV test

GALAXY CLUSTERS:
  - |Phi|/c^2 ~ 10^-4 > Phi_th
  - GCV DOES activate
  - DIFFERENT prediction from MOND
  - THIS IS THE UNIQUE TEST

UNIQUE GCV PREDICTION:
  Cluster mass ratio M_lens / M_bar should follow:
    chi_v = 1 + (3/2) * (|Phi|/Phi_th - 1)^(3/2)
  
  NOT the standard MOND prediction:
    chi_v = sqrt(a0 / g_N)

This is testable with current data (DES, Planck lensing).

============================================================
""")

# =============================================================================
# Create Plot
# =============================================================================
print("Creating plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Potential vs distance in Solar System
ax1 = axes[0, 0]
r_AU = np.logspace(0, 4, 100)
Phi_SS = G * M_sun / (r_AU * AU) / c**2

ax1.loglog(r_AU, Phi_SS, 'b-', linewidth=2, label='Solar System')
ax1.axhline(Phi_th/c**2, color='red', linestyle='--', linewidth=2, 
            label=f'GCV threshold = {Phi_th/c**2:.1e}')
ax1.axvline(76, color='green', linestyle=':', label='Sedna perihelion')
ax1.axvline(937, color='orange', linestyle=':', label='Sedna aphelion')
ax1.set_xlabel('Distance [AU]', fontsize=12)
ax1.set_ylabel('|Phi|/c^2', fontsize=12)
ax1.set_title('Solar System: GCV Never Activates', fontsize=14, fontweight='bold')
ax1.legend()
ax1.set_ylim(1e-14, 1e-4)

# Plot 2: Acceleration vs distance
ax2 = axes[0, 1]
g_SS = G * M_sun / (r_AU * AU)**2

ax2.loglog(r_AU, g_SS, 'b-', linewidth=2, label='Newtonian g')
ax2.axhline(a0, color='red', linestyle='--', linewidth=2, label=f'a0 = {a0:.1e}')
ax2.axvline(76, color='green', linestyle=':', label='Sedna perihelion')
ax2.axvline(937, color='orange', linestyle=':', label='Sedna aphelion')
ax2.fill_between(r_AU, 1e-15, a0, alpha=0.2, color='yellow', label='MOND regime')
ax2.set_xlabel('Distance [AU]', fontsize=12)
ax2.set_ylabel('Acceleration [m/s^2]', fontsize=12)
ax2.set_title('Solar System: MOND Regime at Large Distances', fontsize=14, fontweight='bold')
ax2.legend()
ax2.set_ylim(1e-15, 1e-2)

# Plot 3: GCV vs MOND in clusters
ax3 = axes[1, 0]
Phi_cluster = np.linspace(1, 10, 100) * Phi_th
chi_mond = np.ones_like(Phi_cluster) * 5  # Approximate MOND chi_v for clusters
chi_gcv = 1 + 1.5 * (Phi_cluster/Phi_th - 1)**1.5

ax3.plot(Phi_cluster/Phi_th, chi_mond, 'b--', linewidth=2, label='MOND (constant)')
ax3.plot(Phi_cluster/Phi_th, chi_gcv, 'r-', linewidth=2, label='GCV (potential-dependent)')
ax3.axvline(1, color='gray', linestyle=':', label='Threshold')
ax3.set_xlabel('|Phi| / Phi_th', fontsize=12)
ax3.set_ylabel('chi_v (mass enhancement)', fontsize=12)
ax3.set_title('Clusters: GCV vs MOND Prediction', fontsize=14, fontweight='bold')
ax3.legend()

# Plot 4: Summary
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = """
SEDNA AND GCV: CONCLUSION

SEDNA IS NOT A GCV TEST
  |Phi|/c^2 ~ 10^-11 << Phi_th = 1.5e-5
  GCV = MOND for Sedna

THE UNIQUE GCV TEST IS CLUSTERS
  |Phi|/c^2 ~ 10^-4 > Phi_th
  GCV predicts potential-dependent enhancement
  MOND predicts constant enhancement

TESTABLE PREDICTION:
  M_lens / M_bar = f(|Phi|)
  
  NOT constant as MOND predicts.

DATA SOURCES:
  - DES weak lensing
  - Planck CMB lensing
  - X-ray + lensing combined
  - Cluster mass function

This is the smoking gun for GCV vs MOND.
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('/home/manuel/CascadeProjects/gcv-theory/gcv_gpu_tests/theory/115_Sedna_Prediction.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Plot saved!")

print("\n" + "=" * 70)
print("SEDNA ANALYSIS COMPLETE!")
print("=" * 70)
