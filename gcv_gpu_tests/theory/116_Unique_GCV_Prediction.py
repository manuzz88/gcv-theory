#!/usr/bin/env python3
"""
THE UNIQUE GCV PREDICTION: chi_v vs Phi

This is the SMOKING GUN test for GCV vs MOND.

MOND predicts: chi_v ~ constant (independent of potential)
GCV predicts: chi_v = f(Phi) (increases with potential)

We test this with our 19 clusters.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("=" * 70)
print("THE UNIQUE GCV PREDICTION: chi_v vs Phi")
print("=" * 70)

# =============================================================================
# Constants
# =============================================================================

G = 6.674e-11
c = 3e8
M_sun = 1.989e30
kpc = 3.086e19
Mpc = 3.086e22

f_b = 0.156
Phi_th = (f_b / (2 * np.pi))**3 * c**2
a0 = 1.2e-10

print(f"\nGCV threshold: Phi_th/c^2 = {Phi_th/c**2:.2e}")

# =============================================================================
# Cluster Data (from our previous analysis)
# =============================================================================

# Format: name, M_bar (10^14 M_sun), M_lens (10^14 M_sun), R500 (Mpc)
clusters = [
    # Relaxed clusters
    ("Coma", 1.2, 12.0, 1.4),
    ("Perseus", 0.8, 6.5, 1.2),
    ("Virgo", 0.3, 2.5, 0.8),
    ("A2029", 1.0, 9.0, 1.3),
    ("A478", 0.9, 8.0, 1.2),
    ("A1795", 0.7, 6.0, 1.1),
    ("A2142", 1.1, 10.0, 1.4),
    ("A2256", 0.8, 7.0, 1.2),
    ("A3558", 0.6, 5.0, 1.0),
    ("A3571", 0.7, 6.0, 1.1),
    ("A85", 0.8, 7.0, 1.2),
    ("A1689", 1.5, 15.0, 1.5),
    ("A2218", 0.9, 8.0, 1.2),
    ("RXJ1347", 2.0, 20.0, 1.6),
    # Merger clusters
    ("Bullet", 1.5, 15.0, 1.0),
    ("El Gordo", 2.7, 22.0, 1.4),
    ("MACS J0025", 1.0, 9.0, 1.1),
    ("Abell 520", 1.25, 8.0, 1.2),
    ("MACS J1149", 1.9, 18.0, 1.4),
]

print(f"\nAnalyzing {len(clusters)} clusters...")

# =============================================================================
# Calculate Phi and chi_v for each cluster
# =============================================================================

results = []

for name, M_bar_14, M_lens_14, R500_Mpc in clusters:
    M_bar = M_bar_14 * 1e14 * M_sun
    M_lens = M_lens_14 * 1e14 * M_sun
    R = R500_Mpc * Mpc
    
    # Gravitational potential
    Phi = G * M_lens / R
    Phi_over_c2 = Phi / c**2
    
    # Observed chi_v
    chi_v_obs = M_lens / M_bar
    
    # GCV prediction
    x = Phi / Phi_th
    if x > 1:
        chi_v_gcv = 1 + 1.5 * (x - 1)**1.5
    else:
        chi_v_gcv = 1.0
    
    results.append({
        'name': name,
        'Phi_over_c2': Phi_over_c2,
        'x': x,
        'chi_v_obs': chi_v_obs,
        'chi_v_gcv': chi_v_gcv,
    })

# =============================================================================
# Test the Correlation
# =============================================================================
print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

Phi_values = np.array([r['Phi_over_c2'] for r in results])
x_values = np.array([r['x'] for r in results])
chi_v_obs = np.array([r['chi_v_obs'] for r in results])
chi_v_gcv = np.array([r['chi_v_gcv'] for r in results])

# Correlation between Phi and chi_v_obs
corr_Phi_chi, p_value = stats.pearsonr(Phi_values, chi_v_obs)

print(f"\nCorrelation between |Phi|/c^2 and chi_v_observed:")
print(f"  Pearson r = {corr_Phi_chi:.3f}")
print(f"  p-value = {p_value:.4f}")

if p_value < 0.05:
    print(f"  SIGNIFICANT at 95% level!")
else:
    print(f"  Not significant at 95% level")

# Correlation between x (Phi/Phi_th) and chi_v_obs
corr_x_chi, p_value_x = stats.pearsonr(x_values, chi_v_obs)

print(f"\nCorrelation between |Phi|/Phi_th and chi_v_observed:")
print(f"  Pearson r = {corr_x_chi:.3f}")
print(f"  p-value = {p_value_x:.4f}")

# =============================================================================
# What MOND Would Predict
# =============================================================================
print("\n" + "=" * 70)
print("MOND vs GCV PREDICTION")
print("=" * 70)

print("""
MOND PREDICTION:
  chi_v should be roughly constant for all clusters
  (depends only on g/a0, which is similar for all clusters)
  
  Expected: chi_v ~ 5 +/- 2 for all clusters
  Correlation with Phi: ~0 (no correlation)

GCV PREDICTION:
  chi_v = 1 + (3/2) * (|Phi|/Phi_th - 1)^(3/2)
  
  chi_v should INCREASE with |Phi|
  Correlation with Phi: positive and significant
""")

# Calculate what MOND would predict
# In deep MOND: chi_v ~ sqrt(a0 / g_N)
chi_v_mond = []
for r in results:
    # Approximate g_N at R500
    M_bar = r['chi_v_obs'] * r['Phi_over_c2'] * c**2 / G  # rough
    R = r['Phi_over_c2'] * c**2 / G / (r['chi_v_obs'] * 1e14 * M_sun) * Mpc
    g_N = G * r['chi_v_obs'] * 1e14 * M_sun / (1.2 * Mpc)**2  # approximate
    chi_v_mond.append(np.sqrt(a0 / g_N) if g_N < a0 else 1.0)

chi_v_mond = np.array(chi_v_mond)
chi_v_mond_mean = np.mean(chi_v_obs)  # Use observed mean as MOND "prediction"

print(f"\nMOND prediction (constant): chi_v ~ {chi_v_mond_mean:.1f}")
print(f"GCV prediction: chi_v varies from {np.min(chi_v_gcv):.1f} to {np.max(chi_v_gcv):.1f}")

# =============================================================================
# The Key Test
# =============================================================================
print("\n" + "=" * 70)
print("THE KEY TEST: IS chi_v CORRELATED WITH Phi?")
print("=" * 70)

print(f"\nObserved correlation: r = {corr_Phi_chi:.3f}")

if corr_Phi_chi > 0.3:
    print("\n*** POSITIVE CORRELATION DETECTED! ***")
    print("This SUPPORTS GCV over MOND!")
    print("chi_v increases with potential, as GCV predicts.")
elif corr_Phi_chi < -0.3:
    print("\n*** NEGATIVE CORRELATION DETECTED! ***")
    print("This is unexpected by both theories.")
else:
    print("\n*** NO SIGNIFICANT CORRELATION ***")
    print("This would support MOND (constant chi_v).")

# =============================================================================
# Detailed Results
# =============================================================================
print("\n" + "=" * 70)
print("DETAILED RESULTS")
print("=" * 70)

print(f"\n{'Cluster':<15} {'|Phi|/Phi_th':<12} {'chi_v_obs':<12} {'chi_v_GCV':<12} {'Match':<10}")
print("-" * 65)

for r in sorted(results, key=lambda x: x['x']):
    match = r['chi_v_gcv'] / r['chi_v_obs'] * 100
    print(f"{r['name']:<15} {r['x']:<12.2f} {r['chi_v_obs']:<12.1f} {r['chi_v_gcv']:<12.1f} {match:<10.0f}%")

# =============================================================================
# Linear Regression
# =============================================================================
print("\n" + "=" * 70)
print("LINEAR REGRESSION: chi_v vs |Phi|/Phi_th")
print("=" * 70)

slope, intercept, r_value, p_value_reg, std_err = stats.linregress(x_values, chi_v_obs)

print(f"\nLinear fit: chi_v = {slope:.2f} * (|Phi|/Phi_th) + {intercept:.2f}")
print(f"R^2 = {r_value**2:.3f}")
print(f"p-value = {p_value_reg:.4f}")

# GCV theoretical prediction
# chi_v = 1 + 1.5 * (x - 1)^1.5
# For x >> 1: chi_v ~ 1.5 * x^1.5
# Linearized for small (x-1): chi_v ~ 1 + 1.5 * (x-1) = -0.5 + 1.5*x

print(f"\nGCV theoretical (linearized): chi_v ~ 1.5 * x - 0.5")
print(f"Observed fit: chi_v ~ {slope:.2f} * x + {intercept:.2f}")

if abs(slope - 1.5) < 0.5:
    print("\n*** SLOPE IS CONSISTENT WITH GCV! ***")
else:
    print(f"\n*** SLOPE DIFFERS FROM GCV PREDICTION (1.5) ***")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY: THE UNIQUE GCV PREDICTION")
print("=" * 70)

print(f"""
============================================================
        THE SMOKING GUN TEST FOR GCV
============================================================

QUESTION:
  Does chi_v (mass enhancement) correlate with potential?

MOND PREDICTION:
  No correlation. chi_v ~ constant ~ 5.

GCV PREDICTION:
  Positive correlation. chi_v increases with |Phi|/Phi_th.

OBSERVED:
  Correlation r = {corr_Phi_chi:.3f}
  p-value = {p_value:.4f}
  Slope = {slope:.2f} (GCV predicts ~1.5)

INTERPRETATION:
""")

if corr_Phi_chi > 0.5 and p_value < 0.05:
    print("  STRONG SUPPORT FOR GCV!")
    print("  chi_v correlates with potential as predicted.")
    print("  This is evidence AGAINST constant-a0 MOND.")
elif corr_Phi_chi > 0.3:
    print("  MODERATE SUPPORT FOR GCV.")
    print("  Positive correlation exists but not highly significant.")
    print("  More data needed for definitive conclusion.")
elif abs(corr_Phi_chi) < 0.3:
    print("  INCONCLUSIVE.")
    print("  No strong correlation detected.")
    print("  Could support either MOND or GCV within uncertainties.")
else:
    print("  UNEXPECTED RESULT.")
    print("  Negative correlation not predicted by either theory.")

print("""
============================================================
""")

# =============================================================================
# Create Plot
# =============================================================================
print("Creating plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: chi_v_obs vs |Phi|/Phi_th
ax1 = axes[0, 0]
ax1.scatter(x_values, chi_v_obs, s=100, c='blue', alpha=0.7, label='Observed')

# GCV prediction line
x_line = np.linspace(1, np.max(x_values) * 1.1, 100)
chi_v_line = 1 + 1.5 * (x_line - 1)**1.5
ax1.plot(x_line, chi_v_line, 'r-', linewidth=2, label='GCV prediction')

# MOND prediction (constant)
ax1.axhline(np.mean(chi_v_obs), color='green', linestyle='--', linewidth=2, 
            label=f'MOND (constant = {np.mean(chi_v_obs):.1f})')

# Linear fit
ax1.plot(x_values, slope * x_values + intercept, 'b:', linewidth=1, 
         label=f'Linear fit (r={corr_Phi_chi:.2f})')

ax1.set_xlabel('|Phi| / Phi_th', fontsize=12)
ax1.set_ylabel('chi_v (observed)', fontsize=12)
ax1.set_title('THE KEY TEST: chi_v vs Potential', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: chi_v_obs vs chi_v_GCV
ax2 = axes[0, 1]
ax2.scatter(chi_v_gcv, chi_v_obs, s=100, c='blue', alpha=0.7)
ax2.plot([0, 20], [0, 20], 'r--', linewidth=2, label='Perfect match')
ax2.set_xlabel('chi_v (GCV prediction)', fontsize=12)
ax2.set_ylabel('chi_v (observed)', fontsize=12)
ax2.set_title('GCV Prediction vs Observation', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 20)
ax2.set_ylim(0, 20)

# Plot 3: Residuals
ax3 = axes[1, 0]
residuals_gcv = (chi_v_obs - chi_v_gcv) / chi_v_obs * 100
residuals_mond = (chi_v_obs - np.mean(chi_v_obs)) / chi_v_obs * 100

ax3.scatter(x_values, residuals_gcv, s=100, c='red', alpha=0.7, label='GCV residuals')
ax3.scatter(x_values, residuals_mond, s=100, c='green', alpha=0.7, marker='s', label='MOND residuals')
ax3.axhline(0, color='black', linestyle='-')
ax3.set_xlabel('|Phi| / Phi_th', fontsize=12)
ax3.set_ylabel('Residual (%)', fontsize=12)
ax3.set_title('Residuals: GCV vs MOND', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Summary
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
THE UNIQUE GCV PREDICTION

MOND: chi_v = constant
GCV: chi_v = f(|Phi|)

OBSERVED CORRELATION:
  r = {corr_Phi_chi:.3f}
  p-value = {p_value:.4f}

LINEAR FIT:
  chi_v = {slope:.2f} * x + {intercept:.2f}
  GCV predicts slope ~ 1.5

RMS RESIDUALS:
  GCV: {np.std(residuals_gcv):.1f}%
  MOND: {np.std(residuals_mond):.1f}%

CONCLUSION:
{"GCV SUPPORTED" if corr_Phi_chi > 0.3 else "INCONCLUSIVE"}

This is the smoking gun test.
chi_v should correlate with potential
if GCV is correct.
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen' if corr_Phi_chi > 0.3 else 'lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('/home/manuel/CascadeProjects/gcv-theory/gcv_gpu_tests/theory/116_Unique_GCV_Prediction.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Plot saved!")

print("\n" + "=" * 70)
print("UNIQUE GCV PREDICTION TEST COMPLETE!")
print("=" * 70)
