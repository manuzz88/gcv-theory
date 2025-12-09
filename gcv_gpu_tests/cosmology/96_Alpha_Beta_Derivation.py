#!/usr/bin/env python3
"""
THEORETICAL DERIVATION OF ALPHA AND BETA

We found empirically: alpha = 1.49, beta = 1.46

Can we derive these from first principles?

The enhancement function is:
  a0_eff = a0 * (1 + alpha * (|Phi|/Phi_th - 1)^beta)

What physical meaning could alpha and beta have?
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("THEORETICAL DERIVATION OF ALPHA AND BETA")
print("=" * 70)

# =============================================================================
# Constants
# =============================================================================

G = 6.674e-11
c = 3e8
H0 = 2.2e-18
a0 = 1.2e-10

f_b = 0.156
Phi_th = (f_b / (2 * np.pi))**3 * c**2

# Empirical values
alpha_emp = 1.49
beta_emp = 1.46

print(f"\nEmpirical values from cluster fit:")
print(f"  alpha = {alpha_emp:.2f}")
print(f"  beta = {beta_emp:.2f}")

# =============================================================================
# Observation: beta ~ 1.5 ~ 3/2
# =============================================================================
print("\n" + "=" * 70)
print("OBSERVATION: beta ~ 3/2")
print("=" * 70)

print(f"""
beta = {beta_emp:.2f} is very close to 3/2 = 1.5

In physics, 3/2 appears in:
1. Equipartition theorem: E = (3/2) kT
2. Adiabatic index for monatomic gas: gamma = 5/3, but (gamma-1) = 2/3
3. Gravitational potential energy: U ~ -GM^2/R
4. Virial theorem: 2K + U = 0

Could beta = 3/2 have a physical meaning?

If beta = 3/2, then:
  a0_eff = a0 * (1 + alpha * (|Phi|/Phi_th - 1)^(3/2))

This is a 3/2 power law, which appears in:
- Kepler's third law: T^2 ~ a^3 -> T ~ a^(3/2)
- Density of states in 3D: g(E) ~ E^(1/2) -> integrated: N ~ E^(3/2)
""")

# =============================================================================
# Observation: alpha ~ 1.5 ~ 3/2
# =============================================================================
print("\n" + "=" * 70)
print("OBSERVATION: alpha ~ 3/2")
print("=" * 70)

print(f"""
alpha = {alpha_emp:.2f} is also close to 3/2!

If both alpha = beta = 3/2, then:
  a0_eff = a0 * (1 + (3/2) * (|Phi|/Phi_th - 1)^(3/2))

Let's test this "3/2 hypothesis":
""")

# Test with alpha = beta = 3/2
alpha_theory = 1.5
beta_theory = 1.5

# Cluster data
clusters = {
    "Bullet": {"chi_v_needed": 10.0, "Phi_over_c2": 7.17e-5, "g": 0.17 * a0},
    "Coma": {"chi_v_needed": 5.4, "Phi_over_c2": 2.39e-5, "g": 0.08 * a0},
    "Abell 1689": {"chi_v_needed": 6.0, "Phi_over_c2": 4.78e-5, "g": 0.16 * a0},
    "El Gordo": {"chi_v_needed": 8.8, "Phi_over_c2": 7.01e-5, "g": 0.13 * a0},
}

def chi_v_enhanced(g, Phi_over_c2, alpha, beta):
    Phi = Phi_over_c2 * c**2
    if abs(Phi) <= Phi_th:
        a0_eff = a0
    else:
        x = abs(Phi) / Phi_th
        a0_eff = a0 * (1 + alpha * (x - 1)**beta)
    return 0.5 * (1 + np.sqrt(1 + 4 * a0_eff / g))

print(f"Testing alpha = beta = 3/2:")
print(f"{'Cluster':<15} {'chi_v need':<12} {'chi_v (3/2)':<12} {'chi_v (emp)':<12} {'Match (3/2)':<12}")
print("-" * 65)

for name, data in clusters.items():
    cv_theory = chi_v_enhanced(data["g"], data["Phi_over_c2"], alpha_theory, beta_theory)
    cv_emp = chi_v_enhanced(data["g"], data["Phi_over_c2"], alpha_emp, beta_emp)
    match = cv_theory / data["chi_v_needed"] * 100
    print(f"{name:<15} {data['chi_v_needed']:<12.1f} {cv_theory:<12.1f} {cv_emp:<12.1f} {match:<12.0f}%")

# =============================================================================
# Why 3/2? Physical Derivation Attempt
# =============================================================================
print("\n" + "=" * 70)
print("WHY 3/2? PHYSICAL DERIVATION ATTEMPT")
print("=" * 70)

print("""
============================================================
        DERIVATION ATTEMPT 1: VIRIAL THEOREM
============================================================

In a virialized system:
  2K + U = 0
  K = -U/2

The kinetic energy is:
  K = (1/2) M sigma^2

The potential energy is:
  U = -G M^2 / R ~ -M * |Phi|

So:
  sigma^2 ~ |Phi|

The MOND effect depends on g/a0.
In deep MOND: g_obs ~ sqrt(g_N * a0)

The enhancement should scale with how "deep" we are in the potential.
The natural measure is:
  x = |Phi| / Phi_th

The enhancement function could be:
  f(x) = x^n for some power n

For a 3D system, the natural power is related to the
dimensionality: n = d/2 = 3/2.

This gives:
  a0_eff = a0 * (1 + A * (x - 1)^(3/2))

where A is a dimensionless constant.
""")

print("""
============================================================
        DERIVATION ATTEMPT 2: PHASE SPACE VOLUME
============================================================

The vacuum coherence in GCV depends on the phase space volume.

In 3D, the density of states scales as:
  g(E) ~ E^(1/2)

The integrated number of states up to energy E:
  N(E) ~ E^(3/2)

If the enhancement is proportional to the number of
"coherent modes" in the potential well:
  Enhancement ~ (|Phi|/Phi_th)^(3/2)

This naturally gives beta = 3/2.
""")

print("""
============================================================
        DERIVATION ATTEMPT 3: DIMENSIONAL ANALYSIS
============================================================

The only dimensionless combination involving Phi is:
  x = |Phi| / Phi_th

The enhancement must be a function f(x).

For x >> 1 (deep potential), we expect power-law behavior:
  f(x) ~ x^n

The exponent n should be determined by the physics.

In 3D gravity, the natural exponent is:
  n = (d-1)/2 = (3-1)/2 = 1

But if the effect involves VOLUME (3D), then:
  n = d/2 = 3/2

The factor 3/2 suggests the enhancement is related to
the 3D volume of the coherent region.
""")

# =============================================================================
# Why alpha ~ 3/2?
# =============================================================================
print("\n" + "=" * 70)
print("WHY alpha ~ 3/2?")
print("=" * 70)

print("""
============================================================
        DERIVATION OF alpha
============================================================

The amplitude alpha determines how STRONG the enhancement is.

If alpha = 3/2, then the full formula is:
  a0_eff = a0 * (1 + (3/2) * (x - 1)^(3/2))

At x = 2 (twice the threshold):
  a0_eff = a0 * (1 + 1.5 * 1^1.5) = a0 * 2.5

At x = 5 (five times threshold):
  a0_eff = a0 * (1 + 1.5 * 4^1.5) = a0 * (1 + 12) = 13 * a0

This seems reasonable for cluster scales.

POSSIBLE PHYSICAL MEANING OF alpha = 3/2:

1. Related to the baryon fraction:
   f_b ~ 0.16 ~ 1/6
   But 3/2 != 1/6

2. Related to the GCV phase factor:
   a0 = cH0/(2*pi)
   2*pi ~ 6.28
   3/2 = 1.5

3. Related to the dimensionality:
   In 3D, many quantities have factors of 3/2.

4. Simply a numerical coincidence.

HONEST ASSESSMENT:
We cannot rigorously derive alpha = 3/2.
It's a plausible value, but not proven.
""")

# =============================================================================
# Test the "3/2 Hypothesis"
# =============================================================================
print("\n" + "=" * 70)
print("TESTING THE 3/2 HYPOTHESIS")
print("=" * 70)

# Compare empirical vs theoretical (3/2, 3/2)
print(f"\nComparison: Empirical ({alpha_emp:.2f}, {beta_emp:.2f}) vs Theoretical (1.5, 1.5)")

errors_emp = []
errors_theory = []

for name, data in clusters.items():
    cv_emp = chi_v_enhanced(data["g"], data["Phi_over_c2"], alpha_emp, beta_emp)
    cv_theory = chi_v_enhanced(data["g"], data["Phi_over_c2"], 1.5, 1.5)
    
    err_emp = (cv_emp / data["chi_v_needed"] - 1)**2
    err_theory = (cv_theory / data["chi_v_needed"] - 1)**2
    
    errors_emp.append(err_emp)
    errors_theory.append(err_theory)

rms_emp = np.sqrt(np.mean(errors_emp))
rms_theory = np.sqrt(np.mean(errors_theory))

print(f"\nRMS error (empirical): {rms_emp*100:.1f}%")
print(f"RMS error (3/2, 3/2): {rms_theory*100:.1f}%")

if rms_theory < rms_emp * 1.5:
    print("\nThe 3/2 hypothesis is COMPARABLE to the empirical fit!")
    print("This suggests alpha = beta = 3/2 may have physical meaning.")
else:
    print("\nThe 3/2 hypothesis is WORSE than the empirical fit.")
    print("The values 1.49, 1.46 may just be numerical coincidences.")

# =============================================================================
# Alternative: alpha = beta = pi/2
# =============================================================================
print("\n" + "=" * 70)
print("ALTERNATIVE: alpha = beta = pi/2")
print("=" * 70)

print(f"""
Another possibility: alpha = beta = pi/2 ~ 1.57

This would connect to the GCV phase factor 2*pi.

pi/2 = {np.pi/2:.4f}
Empirical alpha = {alpha_emp:.4f}
Empirical beta = {beta_emp:.4f}

Difference from pi/2:
  alpha: {abs(alpha_emp - np.pi/2):.4f} ({abs(alpha_emp - np.pi/2)/alpha_emp*100:.1f}%)
  beta: {abs(beta_emp - np.pi/2):.4f} ({abs(beta_emp - np.pi/2)/beta_emp*100:.1f}%)
""")

# Test pi/2
print("Testing alpha = beta = pi/2:")
print(f"{'Cluster':<15} {'chi_v need':<12} {'chi_v (pi/2)':<12} {'Match':<12}")
print("-" * 55)

errors_pi2 = []
for name, data in clusters.items():
    cv = chi_v_enhanced(data["g"], data["Phi_over_c2"], np.pi/2, np.pi/2)
    match = cv / data["chi_v_needed"] * 100
    errors_pi2.append((cv / data["chi_v_needed"] - 1)**2)
    print(f"{name:<15} {data['chi_v_needed']:<12.1f} {cv:<12.1f} {match:<12.0f}%")

rms_pi2 = np.sqrt(np.mean(errors_pi2))
print(f"\nRMS error (pi/2, pi/2): {rms_pi2*100:.1f}%")

# =============================================================================
# Summary of Theoretical Candidates
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY OF THEORETICAL CANDIDATES")
print("=" * 70)

candidates = [
    ("Empirical fit", alpha_emp, beta_emp, rms_emp),
    ("3/2, 3/2", 1.5, 1.5, rms_theory),
    ("pi/2, pi/2", np.pi/2, np.pi/2, rms_pi2),
]

print(f"{'Candidate':<20} {'alpha':<10} {'beta':<10} {'RMS Error':<12}")
print("-" * 55)

for name, a, b, rms in candidates:
    print(f"{name:<20} {a:<10.4f} {b:<10.4f} {rms*100:<12.1f}%")

# Find best theoretical candidate
best_theory = min(candidates[1:], key=lambda x: x[3])
print(f"\nBest theoretical candidate: {best_theory[0]}")
print(f"  alpha = {best_theory[1]:.4f}")
print(f"  beta = {best_theory[2]:.4f}")
print(f"  RMS error = {best_theory[3]*100:.1f}%")

# =============================================================================
# Final Assessment
# =============================================================================
print("\n" + "=" * 70)
print("FINAL ASSESSMENT")
print("=" * 70)

print(f"""
============================================================
        THEORETICAL DERIVATION: HONEST ASSESSMENT
============================================================

WHAT WE FOUND:
- Empirical best fit: alpha = {alpha_emp:.2f}, beta = {beta_emp:.2f}
- Both are close to 3/2 = 1.5
- Also close to pi/2 ~ 1.57

THEORETICAL CANDIDATES:
1. alpha = beta = 3/2 (dimensional argument)
2. alpha = beta = pi/2 (GCV phase connection)

BEST THEORETICAL FIT: {best_theory[0]}
  RMS error: {best_theory[3]*100:.1f}% (vs {rms_emp*100:.1f}% empirical)

CAN WE CLAIM DERIVATION?

PARTIALLY.

The value 3/2 has physical motivation:
- Related to 3D phase space
- Appears in virial theorem
- Natural for gravitational systems

But we CANNOT prove that alpha = beta = 3/2 from first principles.
It's a plausible hypothesis, not a derivation.

HONEST STATUS:
- The enhancement function has a PLAUSIBLE form
- The parameters are CLOSE to simple theoretical values
- But this is NOT a rigorous derivation

============================================================
""")

# =============================================================================
# Create Plot
# =============================================================================
print("Creating plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Enhancement functions
ax1 = axes[0, 0]
x_range = np.linspace(1.01, 10, 100)

for (name, a, b, _) in candidates:
    y = 1 + a * (x_range - 1)**b
    ax1.plot(x_range, y, linewidth=2, label=f'{name} (a={a:.2f}, b={b:.2f})')

ax1.set_xlabel('|Phi| / Phi_th', fontsize=12)
ax1.set_ylabel('a0_eff / a0', fontsize=12)
ax1.set_title('Enhancement Functions', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Cluster comparison
ax2 = axes[0, 1]
cluster_names = list(clusters.keys())
chi_needed = [clusters[n]["chi_v_needed"] for n in cluster_names]
chi_emp = [chi_v_enhanced(clusters[n]["g"], clusters[n]["Phi_over_c2"], alpha_emp, beta_emp) for n in cluster_names]
chi_32 = [chi_v_enhanced(clusters[n]["g"], clusters[n]["Phi_over_c2"], 1.5, 1.5) for n in cluster_names]
chi_pi2 = [chi_v_enhanced(clusters[n]["g"], clusters[n]["Phi_over_c2"], np.pi/2, np.pi/2) for n in cluster_names]

x = np.arange(len(cluster_names))
width = 0.2

ax2.bar(x - 1.5*width, chi_needed, width, label='Observed', color='blue', alpha=0.7)
ax2.bar(x - 0.5*width, chi_emp, width, label='Empirical', color='green', alpha=0.7)
ax2.bar(x + 0.5*width, chi_32, width, label='3/2, 3/2', color='orange', alpha=0.7)
ax2.bar(x + 1.5*width, chi_pi2, width, label='pi/2, pi/2', color='red', alpha=0.7)

ax2.set_xticks(x)
ax2.set_xticklabels(cluster_names, rotation=45, ha='right')
ax2.set_ylabel('chi_v', fontsize=12)
ax2.set_title('Cluster chi_v: Different Parameters', fontsize=14, fontweight='bold')
ax2.legend()

# Plot 3: Error comparison
ax3 = axes[1, 0]
names = [c[0] for c in candidates]
errors = [c[3]*100 for c in candidates]

ax3.bar(names, errors, color=['green', 'orange', 'red'], alpha=0.7)
ax3.set_ylabel('RMS Error (%)', fontsize=12)
ax3.set_title('RMS Error by Parameter Choice', fontsize=14, fontweight='bold')
ax3.axhline(10, color='black', linestyle='--', alpha=0.5, label='10% threshold')

# Plot 4: Summary
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
ALPHA AND BETA: THEORETICAL ANALYSIS

Empirical values:
  alpha = {alpha_emp:.2f}
  beta = {beta_emp:.2f}

Theoretical candidates:
  3/2, 3/2: RMS = {rms_theory*100:.1f}%
  pi/2, pi/2: RMS = {rms_pi2*100:.1f}%

Physical motivation for 3/2:
- 3D phase space volume
- Virial theorem connection
- Dimensional analysis

ASSESSMENT:
The values are CLOSE to simple fractions
(3/2 or pi/2), suggesting possible
theoretical origin.

But we CANNOT rigorously derive them.
This is a PLAUSIBLE HYPOTHESIS,
not a proven derivation.

STATUS: PARTIALLY THEORETICAL
The form is motivated, but not proven.
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('/home/manuel/CascadeProjects/gcv-theory/gcv_gpu_tests/cosmology/96_Alpha_Beta_Derivation.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Plot saved!")

print("\n" + "=" * 70)
print("DERIVATION ANALYSIS COMPLETE")
print("=" * 70)
