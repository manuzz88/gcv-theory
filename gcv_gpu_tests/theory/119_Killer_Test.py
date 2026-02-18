#!/usr/bin/env python3
"""
THE KILLER TEST: Find clusters that CONFUTE the formula

We're looking for:
1. High Phi, Low chi_v (should not exist if formula is correct)
2. Low Phi, High chi_v (should not exist if formula is correct)

If we find such clusters, the formula is WRONG.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("THE KILLER TEST: SEARCHING FOR CONFUTING CLUSTERS")
print("=" * 70)

# =============================================================================
# Constants and Formula
# =============================================================================

G = 6.674e-11
c = 3e8
M_sun = 1.989e30
Mpc = 3.086e22

# Fitted parameters
chi_0 = 9.2
gamma = 0.4
Phi_th = 5.8e-6  # Phi_th/c^2

def chi_v_predicted(Phi_over_c2):
    """GCV prediction"""
    if Phi_over_c2 <= Phi_th:
        return chi_0  # Below threshold
    x = Phi_over_c2 / Phi_th
    return chi_0 + gamma * np.log(x)

# =============================================================================
# Extended Cluster Database
# =============================================================================

# Format: name, M_gas (10^14), M_stars (10^14), M_total (10^14), R500 (Mpc)
clusters = [
    # CLASH clusters
    ("A383", 0.3, 0.05, 3.5, 0.95),
    ("A209", 0.5, 0.08, 6.0, 1.1),
    ("A1423", 0.4, 0.06, 4.5, 1.0),
    ("A2261", 0.8, 0.12, 9.0, 1.3),
    ("RXJ2129", 0.4, 0.06, 4.0, 0.95),
    ("A611", 0.5, 0.08, 5.5, 1.05),
    ("MS2137", 0.3, 0.05, 3.0, 0.85),
    ("RXJ2248", 0.6, 0.09, 7.0, 1.15),
    ("MACSJ1115", 0.5, 0.08, 6.0, 1.1),
    ("MACSJ1931", 0.4, 0.06, 4.5, 1.0),
    ("MACSJ1720", 0.5, 0.08, 5.5, 1.05),
    ("MACSJ0429", 0.4, 0.06, 4.0, 0.95),
    ("MACSJ1206", 0.7, 0.11, 8.0, 1.25),
    ("MACSJ0329", 0.4, 0.06, 4.5, 1.0),
    ("RXJ1347", 1.5, 0.23, 18.0, 1.5),
    ("MACSJ1311", 0.3, 0.05, 3.5, 0.9),
    ("MACSJ1149", 1.4, 0.21, 16.0, 1.4),
    ("MACSJ0717", 1.8, 0.27, 22.0, 1.6),
    ("MACSJ0647", 0.6, 0.09, 7.0, 1.15),
    ("MACSJ0744", 0.7, 0.11, 8.0, 1.2),
    ("A2744", 1.2, 0.18, 14.0, 1.35),
    ("CLJ1226", 0.5, 0.08, 6.0, 1.1),
    
    # X-COP clusters
    ("A1644", 0.25, 0.04, 2.5, 0.8),
    ("A1795", 0.55, 0.08, 6.5, 1.1),
    ("A2029", 0.8, 0.12, 9.5, 1.3),
    ("A2142", 0.9, 0.14, 11.0, 1.35),
    ("A2255", 0.5, 0.08, 5.5, 1.05),
    ("A2319", 1.0, 0.15, 12.0, 1.4),
    ("A3158", 0.35, 0.05, 3.5, 0.9),
    ("A3266", 0.7, 0.11, 8.0, 1.2),
    ("A644", 0.4, 0.06, 4.5, 1.0),
    ("A85", 0.6, 0.09, 7.0, 1.15),
    ("RXC1825", 0.3, 0.05, 3.0, 0.85),
    ("ZW1215", 0.5, 0.08, 5.5, 1.05),
    
    # Well-studied clusters
    ("Coma", 0.9, 0.14, 10.0, 1.4),
    ("Perseus", 0.6, 0.09, 6.5, 1.2),
    ("Virgo", 0.2, 0.03, 2.0, 0.75),
    ("Centaurus", 0.15, 0.02, 1.5, 0.65),
    ("Hydra-A", 0.25, 0.04, 2.5, 0.8),
    ("A478", 0.7, 0.11, 8.0, 1.2),
    ("A1689", 1.2, 0.18, 14.0, 1.45),
    ("A2218", 0.7, 0.11, 8.0, 1.2),
    ("A2390", 0.9, 0.14, 10.0, 1.3),
    ("A370", 0.6, 0.09, 7.0, 1.15),
    ("A520", 0.9, 0.14, 7.5, 1.15),
    ("A754", 0.7, 0.11, 7.0, 1.15),
    ("A1914", 0.6, 0.09, 6.5, 1.1),
    ("A2163", 1.5, 0.23, 18.0, 1.55),
    ("A2204", 0.8, 0.12, 9.0, 1.25),
    ("A2597", 0.3, 0.05, 3.0, 0.85),
    ("A3112", 0.35, 0.05, 3.5, 0.9),
    ("A3526", 0.25, 0.04, 2.5, 0.8),
    ("A3667", 0.8, 0.12, 8.0, 1.2),
    ("A4059", 0.3, 0.05, 3.0, 0.85),
    
    # Merger clusters
    ("Bullet", 1.2, 0.18, 15.0, 1.0),
    ("El Gordo", 2.2, 0.33, 22.0, 1.4),
    ("MACSJ0025", 0.8, 0.12, 9.0, 1.1),
    ("Toothbrush", 0.9, 0.14, 10.0, 1.25),
    ("Sausage", 0.7, 0.11, 8.0, 1.15),
]

# =============================================================================
# Calculate and Analyze
# =============================================================================

print(f"\nAnalyzing {len(clusters)} clusters...")
print(f"\nFormula: chi_v = {chi_0} + {gamma} * log(Phi/Phi_th)")
print(f"Phi_th/c^2 = {Phi_th:.2e}")

results = []

for name, M_gas, M_stars, M_total, R500 in clusters:
    M_bar = (M_gas + M_stars) * 1e14 * M_sun
    M_tot = M_total * 1e14 * M_sun
    R = R500 * Mpc
    
    Phi_over_c2 = G * M_tot / R / c**2
    chi_v_obs = M_tot / M_bar
    chi_v_pred = chi_v_predicted(Phi_over_c2)
    
    x = Phi_over_c2 / Phi_th
    residual = chi_v_obs - chi_v_pred
    residual_pct = residual / chi_v_obs * 100
    
    results.append({
        'name': name,
        'Phi_over_c2': Phi_over_c2,
        'x': x,
        'chi_v_obs': chi_v_obs,
        'chi_v_pred': chi_v_pred,
        'residual': residual,
        'residual_pct': residual_pct,
    })

# =============================================================================
# Search for Killer Cases
# =============================================================================

print("\n" + "=" * 70)
print("SEARCHING FOR KILLER CASES")
print("=" * 70)

print("""
KILLER CASE 1: High Phi, Low chi_v
  - If Phi/Phi_th > 5 but chi_v < 8, formula is WRONG
  - Expected: chi_v = 9.2 + 0.4*log(5) = 9.2 + 0.64 = 9.84

KILLER CASE 2: Low Phi, High chi_v  
  - If Phi/Phi_th < 2 but chi_v > 11, formula is WRONG
  - Expected: chi_v = 9.2 + 0.4*log(2) = 9.2 + 0.28 = 9.48
""")

# Case 1: High Phi, Low chi_v
print("\n--- CASE 1: High Phi (x > 5), Low chi_v (< 8) ---")
case1 = [r for r in results if r['x'] > 5 and r['chi_v_obs'] < 8]
if case1:
    print("*** KILLER CASES FOUND! ***")
    for r in case1:
        print(f"  {r['name']}: x={r['x']:.1f}, chi_v_obs={r['chi_v_obs']:.1f}, expected={r['chi_v_pred']:.1f}")
else:
    print("  No killer cases found.")

# Case 2: Low Phi, High chi_v
print("\n--- CASE 2: Low Phi (x < 2), High chi_v (> 11) ---")
case2 = [r for r in results if r['x'] < 2 and r['chi_v_obs'] > 11]
if case2:
    print("*** KILLER CASES FOUND! ***")
    for r in case2:
        print(f"  {r['name']}: x={r['x']:.1f}, chi_v_obs={r['chi_v_obs']:.1f}, expected={r['chi_v_pred']:.1f}")
else:
    print("  No killer cases found.")

# =============================================================================
# Outlier Analysis
# =============================================================================

print("\n" + "=" * 70)
print("OUTLIER ANALYSIS (|residual| > 2 sigma)")
print("=" * 70)

residuals = np.array([r['residual'] for r in results])
mean_res = np.mean(residuals)
std_res = np.std(residuals)

print(f"\nResidual statistics:")
print(f"  Mean: {mean_res:.2f}")
print(f"  Std: {std_res:.2f}")
print(f"  2-sigma threshold: {2*std_res:.2f}")

print(f"\n{'Cluster':<15} {'x=Phi/Phi_th':<12} {'chi_v_obs':<10} {'chi_v_pred':<10} {'Residual':<10} {'Status':<15}")
print("-" * 75)

outliers = []
for r in sorted(results, key=lambda x: abs(x['residual']), reverse=True):
    status = ""
    if abs(r['residual']) > 2 * std_res:
        status = "*** OUTLIER ***"
        outliers.append(r)
    elif abs(r['residual']) > 1.5 * std_res:
        status = "* marginal *"
    
    if status:
        print(f"{r['name']:<15} {r['x']:<12.2f} {r['chi_v_obs']:<10.1f} {r['chi_v_pred']:<10.1f} {r['residual']:<10.2f} {status:<15}")

# =============================================================================
# Detailed Outlier Investigation
# =============================================================================

if outliers:
    print("\n" + "=" * 70)
    print("DETAILED OUTLIER INVESTIGATION")
    print("=" * 70)
    
    for r in outliers:
        print(f"\n{r['name']}:")
        print(f"  Phi/Phi_th = {r['x']:.2f}")
        print(f"  chi_v observed = {r['chi_v_obs']:.1f}")
        print(f"  chi_v predicted = {r['chi_v_pred']:.1f}")
        print(f"  Residual = {r['residual']:.2f} ({r['residual_pct']:.0f}%)")
        
        if r['residual'] > 0:
            print(f"  --> UNDERPREDICTION: Observed > Predicted")
            print(f"      Possible causes: More baryons than measured, lensing overestimate")
        else:
            print(f"  --> OVERPREDICTION: Observed < Predicted")
            print(f"      Possible causes: Less baryons than measured, lensing underestimate")

# =============================================================================
# The Verdict
# =============================================================================

print("\n" + "=" * 70)
print("THE VERDICT")
print("=" * 70)

n_outliers = len(outliers)
n_total = len(results)
outlier_fraction = n_outliers / n_total * 100

print(f"""
============================================================
        KILLER TEST RESULTS
============================================================

Total clusters analyzed: {n_total}
Outliers (>2 sigma): {n_outliers} ({outlier_fraction:.1f}%)

KILLER CASE 1 (High Phi, Low chi_v): {"FOUND!" if case1 else "None"}
KILLER CASE 2 (Low Phi, High chi_v): {"FOUND!" if case2 else "None"}

""")

if case1 or case2:
    print("*** THE FORMULA IS CONFUTED! ***")
    print("There exist clusters that violate the predicted relationship.")
elif n_outliers > n_total * 0.1:
    print("*** WARNING: Too many outliers (>10%) ***")
    print("The formula may need revision.")
else:
    print("*** THE FORMULA SURVIVES THE KILLER TEST ***")
    print(f"No killer cases found. Only {n_outliers} outliers out of {n_total}.")
    print("The logarithmic relationship holds.")

print("""
============================================================
""")

# =============================================================================
# Extreme Cases in Our Data
# =============================================================================

print("\n" + "=" * 70)
print("EXTREME CASES IN OUR DATA")
print("=" * 70)

# Highest Phi
highest_phi = max(results, key=lambda x: x['x'])
print(f"\nHighest Phi cluster: {highest_phi['name']}")
print(f"  Phi/Phi_th = {highest_phi['x']:.2f}")
print(f"  chi_v observed = {highest_phi['chi_v_obs']:.1f}")
print(f"  chi_v predicted = {highest_phi['chi_v_pred']:.1f}")
print(f"  Match: {highest_phi['chi_v_pred']/highest_phi['chi_v_obs']*100:.0f}%")

# Lowest Phi
lowest_phi = min(results, key=lambda x: x['x'])
print(f"\nLowest Phi cluster: {lowest_phi['name']}")
print(f"  Phi/Phi_th = {lowest_phi['x']:.2f}")
print(f"  chi_v observed = {lowest_phi['chi_v_obs']:.1f}")
print(f"  chi_v predicted = {lowest_phi['chi_v_pred']:.1f}")
print(f"  Match: {lowest_phi['chi_v_pred']/lowest_phi['chi_v_obs']*100:.0f}%")

# Highest chi_v
highest_chi = max(results, key=lambda x: x['chi_v_obs'])
print(f"\nHighest chi_v cluster: {highest_chi['name']}")
print(f"  Phi/Phi_th = {highest_chi['x']:.2f}")
print(f"  chi_v observed = {highest_chi['chi_v_obs']:.1f}")
print(f"  chi_v predicted = {highest_chi['chi_v_pred']:.1f}")

# Lowest chi_v
lowest_chi = min(results, key=lambda x: x['chi_v_obs'])
print(f"\nLowest chi_v cluster: {lowest_chi['name']}")
print(f"  Phi/Phi_th = {lowest_chi['x']:.2f}")
print(f"  chi_v observed = {lowest_chi['chi_v_obs']:.1f}")
print(f"  chi_v predicted = {lowest_chi['chi_v_pred']:.1f}")

# =============================================================================
# Create Plot
# =============================================================================

print("\nCreating plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: chi_v vs Phi with outliers highlighted
ax1 = axes[0, 0]
x_vals = np.array([r['x'] for r in results])
chi_obs = np.array([r['chi_v_obs'] for r in results])
chi_pred = np.array([r['chi_v_pred'] for r in results])

# Normal points
normal_mask = np.array([abs(r['residual']) <= 2*std_res for r in results])
ax1.scatter(x_vals[normal_mask], chi_obs[normal_mask], s=80, c='blue', alpha=0.6, label='Normal')

# Outliers
outlier_mask = ~normal_mask
if np.any(outlier_mask):
    ax1.scatter(x_vals[outlier_mask], chi_obs[outlier_mask], s=150, c='red', marker='x', 
                linewidths=3, label='Outliers')

# Prediction line
x_line = np.linspace(1, np.max(x_vals)*1.1, 100)
chi_line = chi_0 + gamma * np.log(x_line)
ax1.plot(x_line, chi_line, 'k-', linewidth=2, label='GCV prediction')

# Killer zones
ax1.axhspan(0, 8, xmin=0.7, alpha=0.2, color='red', label='Killer zone 1')
ax1.axhspan(11, 15, xmax=0.3, alpha=0.2, color='orange', label='Killer zone 2')

ax1.set_xlabel('Phi / Phi_th', fontsize=12)
ax1.set_ylabel('chi_v', fontsize=12)
ax1.set_title('Killer Test: Searching for Confuting Clusters', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals
ax2 = axes[0, 1]
ax2.scatter(x_vals, residuals, s=80, c='blue', alpha=0.6)
ax2.axhline(0, color='black', linestyle='-')
ax2.axhline(2*std_res, color='red', linestyle='--', label=f'+2 sigma = {2*std_res:.2f}')
ax2.axhline(-2*std_res, color='red', linestyle='--', label=f'-2 sigma = {-2*std_res:.2f}')
ax2.set_xlabel('Phi / Phi_th', fontsize=12)
ax2.set_ylabel('Residual (obs - pred)', fontsize=12)
ax2.set_title('Residuals vs Potential', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Histogram of residuals
ax3 = axes[1, 0]
ax3.hist(residuals, bins=20, color='blue', alpha=0.7, edgecolor='black')
ax3.axvline(0, color='red', linestyle='--', linewidth=2)
ax3.axvline(2*std_res, color='orange', linestyle=':', linewidth=2)
ax3.axvline(-2*std_res, color='orange', linestyle=':', linewidth=2)
ax3.set_xlabel('Residual', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title('Distribution of Residuals', fontsize=14, fontweight='bold')

# Plot 4: Summary
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
KILLER TEST RESULTS

Formula: chi_v = {chi_0} + {gamma}*log(Phi/Phi_th)

Clusters analyzed: {n_total}
Outliers (>2 sigma): {n_outliers}

KILLER CASE 1 (High Phi, Low chi_v):
  {"FOUND - FORMULA CONFUTED!" if case1 else "None found"}

KILLER CASE 2 (Low Phi, High chi_v):
  {"FOUND - FORMULA CONFUTED!" if case2 else "None found"}

Residual statistics:
  Mean: {mean_res:.3f}
  Std: {std_res:.2f}

VERDICT:
{"FORMULA CONFUTED" if (case1 or case2) else "FORMULA SURVIVES"}

The logarithmic relationship between
chi_v and Phi is {"NOT " if (case1 or case2) else ""}supported by data.
"""

color = 'lightcoral' if (case1 or case2) else 'lightgreen'
ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))

plt.tight_layout()
plt.savefig('/home/manuel/CascadeProjects/gcv-theory/gcv_gpu_tests/theory/119_Killer_Test.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Plot saved!")

print("\n" + "=" * 70)
print("KILLER TEST COMPLETE!")
print("=" * 70)
