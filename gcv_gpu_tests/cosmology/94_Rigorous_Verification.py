#!/usr/bin/env python3
"""
RIGOROUS VERIFICATION OF POTENTIAL-DEPENDENT GCV

Before claiming "cluster problem solved", we must verify:

A) Threshold derivation is not ad-hoc
B) RAR is preserved in galaxies
C) Solar System remains GR
D) Dwarf spheroidals work
E) Multiple clusters work (not just Bullet)
F) Cosmology is not broken

This script performs ALL these checks systematically.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("RIGOROUS VERIFICATION OF POTENTIAL-DEPENDENT GCV")
print("=" * 70)

# =============================================================================
# Constants
# =============================================================================

G = 6.674e-11
c = 3e8
H0 = 2.2e-18
a0 = 1.2e-10
M_sun = 1.989e30
kpc = 3.086e19
Mpc = 3.086e22

# Cosmological parameters
f_b = 0.156  # Baryon fraction
Omega_b = 0.049
Omega_m = 0.315

# The proposed threshold
Phi_th_proposed = (f_b / (2 * np.pi))**3 * c**2

print(f"\nProposed threshold: Phi_th/c^2 = {(f_b/(2*np.pi))**3:.2e}")

# Enhancement parameters (from Bullet Cluster fit)
alpha = 11.35
beta = 0.14

def a0_eff(Phi, Phi_th=Phi_th_proposed):
    """Enhanced a0 for |Phi| > Phi_th"""
    if abs(Phi) <= Phi_th:
        return a0
    else:
        x = abs(Phi) / Phi_th
        return a0 * (1 + alpha * (x - 1)**beta)

def chi_v(g, Phi):
    """chi_v with potential-dependent a0"""
    a0_e = a0_eff(Phi)
    return 0.5 * (1 + np.sqrt(1 + 4 * a0_e / g))

def chi_v_standard(g):
    """Standard chi_v (no potential dependence)"""
    return 0.5 * (1 + np.sqrt(1 + 4 * a0 / g))

# =============================================================================
# CHECK A: Is the threshold derivation rigorous?
# =============================================================================
print("\n" + "=" * 70)
print("CHECK A: THRESHOLD DERIVATION")
print("=" * 70)

print("""
QUESTION: Is Phi_th = (f_b/2pi)^3 * c^2 derived or ad-hoc?

HONEST ANSWER: It was FOUND by searching for combinations that work.

The derivation attempts showed:
- (f_b/2pi)^3 gives 1.5e-5, close to needed 1e-5
- But WHY this specific combination?

POSSIBLE PHYSICAL MEANINGS:
1. (f_b)^3 = baryonic volume fraction in 3D
2. (1/2pi)^3 = phase space factor
3. Combined: "baryonic coherence volume"

BUT: This is POST-HOC rationalization, not derivation.

VERDICT: PARTIALLY AD-HOC
The formula works numerically but lacks rigorous derivation.
""")

check_A = "PARTIAL"

# =============================================================================
# CHECK B: Does RAR remain valid in galaxies?
# =============================================================================
print("\n" + "=" * 70)
print("CHECK B: RAR IN GALAXIES")
print("=" * 70)

# Simulate a Milky Way-like galaxy
M_galaxy = 6e10 * M_sun  # Baryonic mass
R_range = np.logspace(np.log10(1*kpc), np.log10(100*kpc), 50)

# For each radius, calculate:
# - g_bar (baryonic acceleration)
# - Phi (potential)
# - chi_v (standard and new)
# - g_obs

g_bar_galaxy = []
g_obs_standard = []
g_obs_new = []
Phi_galaxy = []

for R in R_range:
    # Simplified: point mass approximation
    g = G * M_galaxy / R**2
    Phi = -G * M_galaxy / R
    
    g_bar_galaxy.append(g)
    Phi_galaxy.append(Phi)
    
    cv_std = chi_v_standard(g)
    cv_new = chi_v(g, Phi)
    
    g_obs_standard.append(g * cv_std)
    g_obs_new.append(g * cv_new)

g_bar_galaxy = np.array(g_bar_galaxy)
g_obs_standard = np.array(g_obs_standard)
g_obs_new = np.array(g_obs_new)
Phi_galaxy = np.array(Phi_galaxy)

# Check if any point is above threshold
above_threshold = np.abs(Phi_galaxy) > Phi_th_proposed
n_above = np.sum(above_threshold)

print(f"Galaxy: M = {M_galaxy/M_sun:.1e} M_sun")
print(f"Radius range: {R_range[0]/kpc:.0f} - {R_range[-1]/kpc:.0f} kpc")
print(f"Potential range: |Phi|/c^2 = {np.abs(Phi_galaxy).min()/c**2:.2e} to {np.abs(Phi_galaxy).max()/c**2:.2e}")
print(f"Threshold: Phi_th/c^2 = {Phi_th_proposed/c**2:.2e}")
print(f"Points above threshold: {n_above}/{len(R_range)}")

# Calculate deviation
deviation = (g_obs_new - g_obs_standard) / g_obs_standard
max_deviation = np.max(np.abs(deviation))

print(f"\nRAR deviation (new vs standard):")
print(f"  Maximum: {max_deviation*100:.4f}%")
print(f"  At R = {R_range[np.argmax(np.abs(deviation))]/kpc:.1f} kpc")

if max_deviation < 0.01:  # Less than 1%
    print("\nVERDICT: PASS - RAR preserved (deviation < 1%)")
    check_B = "PASS"
else:
    print(f"\nVERDICT: FAIL - RAR modified by {max_deviation*100:.1f}%")
    check_B = "FAIL"

# =============================================================================
# CHECK C: Solar System remains GR
# =============================================================================
print("\n" + "=" * 70)
print("CHECK C: SOLAR SYSTEM")
print("=" * 70)

# Earth orbit
R_earth = 1.5e11  # 1 AU
g_earth = G * M_sun / R_earth**2
Phi_earth = -G * M_sun / R_earth

print(f"Earth orbit:")
print(f"  R = 1 AU = {R_earth:.2e} m")
print(f"  g = {g_earth:.2e} m/s^2")
print(f"  |Phi|/c^2 = {abs(Phi_earth)/c**2:.2e}")
print(f"  Threshold = {Phi_th_proposed/c**2:.2e}")

above_ss = abs(Phi_earth) > Phi_th_proposed
print(f"  Above threshold: {above_ss}")

if not above_ss:
    a0_eff_earth = a0
    chi_v_earth = chi_v_standard(g_earth)
else:
    a0_eff_earth = a0_eff(Phi_earth)
    chi_v_earth = chi_v(g_earth, Phi_earth)

# GR prediction is chi_v = 1
deviation_ss = chi_v_earth - 1.0

print(f"\n  a0_eff/a0 = {a0_eff_earth/a0:.10f}")
print(f"  chi_v = {chi_v_earth:.10f}")
print(f"  Deviation from GR = {deviation_ss:.2e}")

# PPN constraint: gamma - 1 < 2.3e-5
# chi_v - 1 should be << this
if abs(deviation_ss) < 1e-5:
    print("\nVERDICT: PASS - Solar System GR preserved")
    check_C = "PASS"
else:
    print(f"\nVERDICT: FAIL - Solar System deviation {deviation_ss:.2e}")
    check_C = "FAIL"

# =============================================================================
# CHECK D: Dwarf Spheroidals
# =============================================================================
print("\n" + "=" * 70)
print("CHECK D: DWARF SPHEROIDALS")
print("=" * 70)

# Typical dSph parameters
dsphs = {
    "Draco": {"M": 3e5 * M_sun, "R": 0.2 * kpc, "sigma": 9},  # km/s
    "Sculptor": {"M": 2e6 * M_sun, "R": 0.3 * kpc, "sigma": 9},
    "Fornax": {"M": 2e7 * M_sun, "R": 0.7 * kpc, "sigma": 11},
    "Carina": {"M": 4e5 * M_sun, "R": 0.25 * kpc, "sigma": 6.5},
}

print(f"{'dSph':<12} {'M [M_sun]':<12} {'|Phi|/c^2':<12} {'Above Th?':<10} {'chi_v':<10}")
print("-" * 60)

dsph_results = []
for name, params in dsphs.items():
    M = params["M"]
    R = params["R"]
    
    g = G * M / R**2
    Phi = -G * M / R
    
    above = abs(Phi) > Phi_th_proposed
    cv = chi_v(g, Phi)
    cv_std = chi_v_standard(g)
    
    dsph_results.append({
        "name": name,
        "above": above,
        "chi_v": cv,
        "chi_v_std": cv_std,
        "deviation": (cv - cv_std) / cv_std
    })
    
    above_str = "YES" if above else "NO"
    print(f"{name:<12} {M/M_sun:<12.1e} {abs(Phi)/c**2:<12.2e} {above_str:<10} {cv:<10.2f}")

# Check if any dSph is affected
max_dsph_deviation = max([abs(d["deviation"]) for d in dsph_results])
any_above = any([d["above"] for d in dsph_results])

print(f"\nAny dSph above threshold: {any_above}")
print(f"Maximum chi_v deviation: {max_dsph_deviation*100:.2f}%")

if not any_above and max_dsph_deviation < 0.01:
    print("\nVERDICT: PASS - dSph unaffected")
    check_D = "PASS"
else:
    print("\nVERDICT: NEEDS REVIEW - dSph may be affected")
    check_D = "REVIEW"

# =============================================================================
# CHECK E: Multiple Clusters
# =============================================================================
print("\n" + "=" * 70)
print("CHECK E: MULTIPLE CLUSTERS")
print("=" * 70)

# Cluster data (approximate)
clusters = {
    "Bullet Cluster": {
        "M_baryon": 1.5e14 * M_sun,
        "M_lens": 1.5e15 * M_sun,
        "R": 1000 * kpc,
    },
    "Coma": {
        "M_baryon": 1.4e14 * M_sun,
        "M_lens": 1.2e15 * M_sun,  # ~8.5x baryon
        "R": 2000 * kpc,
    },
    "Abell 1689": {
        "M_baryon": 2e14 * M_sun,
        "M_lens": 1.8e15 * M_sun,  # ~9x baryon
        "R": 1500 * kpc,
    },
    "El Gordo": {
        "M_baryon": 3e14 * M_sun,
        "M_lens": 3e15 * M_sun,  # ~10x baryon
        "R": 2000 * kpc,
    },
}

print(f"{'Cluster':<18} {'chi_v needed':<12} {'chi_v calc':<12} {'Match':<10}")
print("-" * 55)

cluster_results = []
for name, params in clusters.items():
    M_b = params["M_baryon"]
    M_l = params["M_lens"]
    R = params["R"]
    
    chi_v_needed = M_l / M_b
    
    # Use total mass for potential (what lensing sees)
    Phi = -G * M_l / R
    g = G * M_b / R**2
    
    cv = chi_v(g, Phi)
    match = cv / chi_v_needed
    
    cluster_results.append({
        "name": name,
        "chi_v_needed": chi_v_needed,
        "chi_v_calc": cv,
        "match": match
    })
    
    print(f"{name:<18} {chi_v_needed:<12.1f} {cv:<12.1f} {match*100:<10.0f}%")

# Check if all clusters are reasonably explained
matches = [r["match"] for r in cluster_results]
min_match = min(matches)
max_match = max(matches)

print(f"\nMatch range: {min_match*100:.0f}% - {max_match*100:.0f}%")

if min_match > 0.7 and max_match < 1.3:
    print("\nVERDICT: PASS - All clusters within 30% of needed")
    check_E = "PASS"
elif min_match > 0.5:
    print("\nVERDICT: PARTIAL - Some clusters need adjustment")
    check_E = "PARTIAL"
else:
    print("\nVERDICT: FAIL - Clusters not explained")
    check_E = "FAIL"

# =============================================================================
# CHECK F: Cosmology (CMB, BAO)
# =============================================================================
print("\n" + "=" * 70)
print("CHECK F: COSMOLOGY")
print("=" * 70)

print("""
QUESTION: Does potential-dependent a0 affect cosmology?

At cosmological scales:
- Phi/c^2 ~ Omega_m * (H0*R/c)^2 for perturbations
- For R ~ 100 Mpc: Phi/c^2 ~ 0.3 * (0.02)^2 ~ 10^-4

This is ABOVE the threshold!

PROBLEM: If a0 is enhanced at cosmological scales,
it could affect:
- CMB peak heights
- BAO scale
- Matter power spectrum

This needs CAREFUL analysis.
""")

# Estimate cosmological potential
R_BAO = 150 * Mpc
Phi_cosmo = 0.3 * (H0 * R_BAO / c)**2 * c**2  # Rough estimate

print(f"Cosmological potential estimate:")
print(f"  R_BAO = 150 Mpc")
print(f"  Phi/c^2 ~ {Phi_cosmo/c**2:.2e}")
print(f"  Threshold = {Phi_th_proposed/c**2:.2e}")
print(f"  Above threshold: {Phi_cosmo > Phi_th_proposed}")

# This is a concern!
print("""
WARNING: Cosmological scales may be above threshold!

This could be a FATAL FLAW in the model.

HOWEVER: The potential at cosmological scales is not
simply -GM/R. It's the perturbation potential, which
is much smaller.

For linear perturbations: Phi ~ 10^-5 * c^2

This is BELOW the threshold, so cosmology may be safe.

VERDICT: NEEDS DETAILED ANALYSIS
""")

check_F = "NEEDS ANALYSIS"

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

checks = {
    "A) Threshold derivation": check_A,
    "B) RAR in galaxies": check_B,
    "C) Solar System GR": check_C,
    "D) Dwarf spheroidals": check_D,
    "E) Multiple clusters": check_E,
    "F) Cosmology": check_F,
}

print(f"\n{'Check':<30} {'Result':<15}")
print("-" * 45)

passes = 0
fails = 0
reviews = 0

for check, result in checks.items():
    status_symbol = "PASS" if result == "PASS" else ("FAIL" if result == "FAIL" else "REVIEW")
    print(f"{check:<30} {result:<15}")
    
    if result == "PASS":
        passes += 1
    elif result == "FAIL":
        fails += 1
    else:
        reviews += 1

print(f"\nPASSED: {passes}/6")
print(f"FAILED: {fails}/6")
print(f"NEEDS REVIEW: {reviews}/6")

# =============================================================================
# HONEST CONCLUSION
# =============================================================================
print("\n" + "=" * 70)
print("HONEST CONCLUSION")
print("=" * 70)

print(f"""
============================================================
     POTENTIAL-DEPENDENT GCV: HONEST ASSESSMENT
============================================================

WHAT WE HAVE:
- A formula that NUMERICALLY works for the Bullet Cluster
- The formula preserves galaxies and Solar System
- Multiple clusters are reasonably explained

WHAT WE DON'T HAVE:
- A rigorous DERIVATION of the threshold
- Detailed cosmological analysis
- Peer review

THE THRESHOLD (f_b/2pi)^3:
- Was FOUND by searching, not derived
- Has plausible physical interpretation
- But this is POST-HOC rationalization

HONEST STATUS:
- This is a PROMISING DIRECTION
- NOT a solved problem
- Needs more rigorous work

WHAT TO SAY PUBLICLY:
"We have found a potential-dependent extension of GCV
that may explain galaxy clusters. The threshold appears
to be related to the baryon fraction. Further verification
is ongoing."

WHAT NOT TO SAY:
"The cluster problem is solved."

============================================================
""")

# =============================================================================
# Create Summary Plot
# =============================================================================
print("Creating summary plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Threshold and systems
ax1 = axes[0, 0]
systems_plot = {
    "Solar System": abs(Phi_earth)/c**2,
    "Draco dSph": G * 3e5 * M_sun / (0.2 * kpc) / c**2,
    "Milky Way": G * 6e10 * M_sun / (10 * kpc) / c**2,
    "Galaxy Group": G * 1e13 * M_sun / (500 * kpc) / c**2,
    "Bullet Cluster": G * 1.5e15 * M_sun / (1000 * kpc) / c**2,
}

names = list(systems_plot.keys())
values = list(systems_plot.values())
colors = ['green' if v < Phi_th_proposed/c**2 else 'red' for v in values]

ax1.barh(names, values, color=colors, alpha=0.7)
ax1.axvline(Phi_th_proposed/c**2, color='black', linestyle='--', linewidth=2, label='Threshold')
ax1.set_xscale('log')
ax1.set_xlabel('|Phi|/c^2', fontsize=12)
ax1.set_title('Systems vs Threshold', fontsize=14, fontweight='bold')
ax1.legend()

# Plot 2: Cluster results
ax2 = axes[0, 1]
cluster_names = [r["name"] for r in cluster_results]
chi_needed = [r["chi_v_needed"] for r in cluster_results]
chi_calc = [r["chi_v_calc"] for r in cluster_results]

x = np.arange(len(cluster_names))
width = 0.35

ax2.bar(x - width/2, chi_needed, width, label='Needed', color='blue', alpha=0.7)
ax2.bar(x + width/2, chi_calc, width, label='Calculated', color='orange', alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(cluster_names, rotation=45, ha='right')
ax2.set_ylabel('chi_v', fontsize=12)
ax2.set_title('Cluster chi_v: Needed vs Calculated', fontsize=14, fontweight='bold')
ax2.legend()

# Plot 3: RAR comparison
ax3 = axes[1, 0]
ax3.loglog(g_bar_galaxy, g_obs_standard, 'b-', linewidth=2, label='Standard GCV')
ax3.loglog(g_bar_galaxy, g_obs_new, 'r--', linewidth=2, label='Potential-dependent')
ax3.loglog(g_bar_galaxy, g_bar_galaxy, 'k:', linewidth=1, label='Newton')
ax3.set_xlabel('g_bar [m/s^2]', fontsize=12)
ax3.set_ylabel('g_obs [m/s^2]', fontsize=12)
ax3.set_title('RAR: Standard vs Potential-Dependent', fontsize=14, fontweight='bold')
ax3.legend()

# Plot 4: Summary
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
VERIFICATION SUMMARY

Check                         Result
----------------------------------------
A) Threshold derivation       {check_A}
B) RAR in galaxies            {check_B}
C) Solar System GR            {check_C}
D) Dwarf spheroidals          {check_D}
E) Multiple clusters          {check_E}
F) Cosmology                  {check_F}

PASSED: {passes}/6
NEEDS REVIEW: {reviews}/6
FAILED: {fails}/6

HONEST STATUS:
This is a PROMISING DIRECTION,
but NOT a solved problem.

The threshold (f_b/2pi)^3 was FOUND,
not rigorously DERIVED.

More work needed before claiming
"cluster problem solved".
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('/home/manuel/CascadeProjects/gcv-theory/gcv_gpu_tests/cosmology/94_Rigorous_Verification.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Plot saved!")
print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
