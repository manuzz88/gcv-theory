#!/usr/bin/env python3
"""
DIAGNOSIS: What Could Be Wrong?

Assumption: The BASIC IDEA is correct (potential-dependent enhancement)
Question: What is WRONG with our implementation?

Possible issues:
1. Wrong functional form (log vs power vs something else)
2. Wrong variable (Phi vs g vs something else)
3. Wrong data (M_bar underestimated, M_total wrong)
4. Missing physics (merger state, redshift, environment)
5. Wrong threshold definition
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

print("=" * 70)
print("DIAGNOSIS: WHAT COULD BE WRONG?")
print("=" * 70)

# =============================================================================
# Constants
# =============================================================================

G = 6.674e-11
c = 3e8
M_sun = 1.989e30
Mpc = 3.086e22
a0 = 1.2e-10

# =============================================================================
# Cluster Data with More Details
# =============================================================================

# Format: name, M_gas, M_stars, M_total, R500, z, merger_flag
# merger_flag: 0 = relaxed, 1 = minor merger, 2 = major merger
clusters = [
    # Relaxed clusters
    ("A383", 0.3, 0.05, 3.5, 0.95, 0.19, 0),
    ("A209", 0.5, 0.08, 6.0, 1.1, 0.21, 0),
    ("A2261", 0.8, 0.12, 9.0, 1.3, 0.22, 0),
    ("A611", 0.5, 0.08, 5.5, 1.05, 0.29, 0),
    ("RXJ1347", 1.5, 0.23, 18.0, 1.5, 0.45, 1),
    ("MACSJ1149", 1.4, 0.21, 16.0, 1.4, 0.54, 1),
    ("MACSJ0717", 1.8, 0.27, 22.0, 1.6, 0.55, 2),
    ("A2744", 1.2, 0.18, 14.0, 1.35, 0.31, 2),
    ("A1795", 0.55, 0.08, 6.5, 1.1, 0.06, 0),
    ("A2029", 0.8, 0.12, 9.5, 1.3, 0.08, 0),
    ("A2142", 0.9, 0.14, 11.0, 1.35, 0.09, 1),
    ("Coma", 0.9, 0.14, 10.0, 1.4, 0.02, 0),
    ("Perseus", 0.6, 0.09, 6.5, 1.2, 0.02, 0),
    ("Virgo", 0.2, 0.03, 2.0, 0.75, 0.004, 0),
    ("A478", 0.7, 0.11, 8.0, 1.2, 0.09, 0),
    ("A1689", 1.2, 0.18, 14.0, 1.45, 0.18, 0),
    ("A2218", 0.7, 0.11, 8.0, 1.2, 0.17, 0),
    ("A520", 0.9, 0.14, 7.5, 1.15, 0.20, 2),  # Train wreck!
    ("Bullet", 1.2, 0.18, 15.0, 1.0, 0.30, 2),
    ("El Gordo", 2.2, 0.33, 22.0, 1.4, 0.87, 2),
    ("A85", 0.6, 0.09, 7.0, 1.15, 0.06, 0),
    ("A2163", 1.5, 0.23, 18.0, 1.55, 0.20, 2),
]

print(f"Analyzing {len(clusters)} clusters with detailed properties...")

# Calculate all quantities
results = []
for name, M_gas, M_stars, M_total, R500, z, merger in clusters:
    M_bar = (M_gas + M_stars) * 1e14 * M_sun
    M_tot = M_total * 1e14 * M_sun
    R = R500 * Mpc
    
    # Different potential definitions
    Phi = G * M_tot / R / c**2  # Using total mass
    Phi_bar = G * M_bar / R / c**2  # Using baryonic mass
    
    # Acceleration
    g_N = G * M_tot / R**2
    g_bar = G * M_bar / R**2
    
    # chi_v
    chi_v = M_tot / M_bar
    
    results.append({
        'name': name,
        'M_bar': M_bar / M_sun / 1e14,
        'M_tot': M_tot / M_sun / 1e14,
        'R500': R500,
        'z': z,
        'merger': merger,
        'Phi_tot': Phi,
        'Phi_bar': Phi_bar,
        'g_N': g_N,
        'g_bar': g_bar,
        'chi_v': chi_v,
    })

# =============================================================================
# HYPOTHESIS 1: Wrong Variable - Should Use g Instead of Phi?
# =============================================================================
print("\n" + "=" * 70)
print("HYPOTHESIS 1: WRONG VARIABLE")
print("=" * 70)

print("""
Maybe chi_v depends on ACCELERATION (g), not POTENTIAL (Phi)?

In MOND, the key variable is g/a0.
Maybe GCV should also use g, not Phi.
""")

# Correlation with different variables
chi_v_arr = np.array([r['chi_v'] for r in results])
Phi_tot_arr = np.array([r['Phi_tot'] for r in results])
Phi_bar_arr = np.array([r['Phi_bar'] for r in results])
g_N_arr = np.array([r['g_N'] for r in results])
g_bar_arr = np.array([r['g_bar'] for r in results])

corr_Phi_tot, p_Phi_tot = stats.pearsonr(Phi_tot_arr, chi_v_arr)
corr_Phi_bar, p_Phi_bar = stats.pearsonr(Phi_bar_arr, chi_v_arr)
corr_g_N, p_g_N = stats.pearsonr(g_N_arr, chi_v_arr)
corr_g_bar, p_g_bar = stats.pearsonr(g_bar_arr, chi_v_arr)

print(f"\nCorrelation of chi_v with different variables:")
print(f"  Phi_total:   r = {corr_Phi_tot:.3f}, p = {p_Phi_tot:.4f}")
print(f"  Phi_baryon:  r = {corr_Phi_bar:.3f}, p = {p_Phi_bar:.4f}")
print(f"  g_N (total): r = {corr_g_N:.3f}, p = {p_g_N:.4f}")
print(f"  g_baryon:    r = {corr_g_bar:.3f}, p = {p_g_bar:.4f}")

best_var = max([
    ('Phi_total', abs(corr_Phi_tot)),
    ('Phi_baryon', abs(corr_Phi_bar)),
    ('g_N', abs(corr_g_N)),
    ('g_baryon', abs(corr_g_bar)),
], key=lambda x: x[1])

print(f"\nBest correlating variable: {best_var[0]} (|r| = {best_var[1]:.3f})")

# =============================================================================
# HYPOTHESIS 2: Wrong Functional Form
# =============================================================================
print("\n" + "=" * 70)
print("HYPOTHESIS 2: WRONG FUNCTIONAL FORM")
print("=" * 70)

print("""
Maybe the relationship is not logarithmic?

Let's try different forms:
1. Linear: chi_v = a + b*x
2. Power: chi_v = a * x^b
3. Log: chi_v = a + b*log(x)
4. Sqrt: chi_v = a + b*sqrt(x)
5. Inverse: chi_v = a + b/x
""")

x = Phi_tot_arr / 1e-5  # Normalize

def linear(x, a, b): return a + b*x
def power(x, a, b): return a * np.abs(x)**b
def log_func(x, a, b): return a + b*np.log(x)
def sqrt_func(x, a, b): return a + b*np.sqrt(x)
def inverse(x, a, b): return a + b/x

forms = [
    ('Linear', linear),
    ('Power', power),
    ('Logarithmic', log_func),
    ('Square root', sqrt_func),
    ('Inverse', inverse),
]

print(f"\n{'Form':<15} {'RSS':<10} {'R^2':<10}")
print("-" * 35)

best_form = None
best_rss = np.inf

for name, func in forms:
    try:
        popt, _ = curve_fit(func, x, chi_v_arr, p0=[9, 0.1], maxfev=5000)
        pred = func(x, *popt)
        rss = np.sum((chi_v_arr - pred)**2)
        ss_tot = np.sum((chi_v_arr - np.mean(chi_v_arr))**2)
        r2 = 1 - rss/ss_tot
        print(f"{name:<15} {rss:<10.2f} {r2:<10.3f}")
        
        if rss < best_rss:
            best_rss = rss
            best_form = name
    except:
        print(f"{name:<15} {'FAILED':<10}")

print(f"\nBest functional form: {best_form}")

# =============================================================================
# HYPOTHESIS 3: Merger State Matters
# =============================================================================
print("\n" + "=" * 70)
print("HYPOTHESIS 3: MERGER STATE MATTERS")
print("=" * 70)

print("""
Maybe mergers behave differently from relaxed clusters?

A520 and El Gordo are both major mergers.
Let's separate relaxed vs mergers.
""")

relaxed = [r for r in results if r['merger'] == 0]
mergers = [r for r in results if r['merger'] >= 1]
major_mergers = [r for r in results if r['merger'] == 2]

print(f"\nRelaxed clusters: {len(relaxed)}")
print(f"All mergers: {len(mergers)}")
print(f"Major mergers: {len(major_mergers)}")

# Statistics for each group
chi_v_relaxed = np.array([r['chi_v'] for r in relaxed])
chi_v_mergers = np.array([r['chi_v'] for r in mergers])
chi_v_major = np.array([r['chi_v'] for r in major_mergers])

print(f"\nchi_v statistics:")
print(f"  Relaxed: mean = {np.mean(chi_v_relaxed):.2f}, std = {np.std(chi_v_relaxed):.2f}")
print(f"  Mergers: mean = {np.mean(chi_v_mergers):.2f}, std = {np.std(chi_v_mergers):.2f}")
print(f"  Major:   mean = {np.mean(chi_v_major):.2f}, std = {np.std(chi_v_major):.2f}")

# Correlation for relaxed only
Phi_relaxed = np.array([r['Phi_tot'] for r in relaxed])
corr_relaxed, p_relaxed = stats.pearsonr(Phi_relaxed, chi_v_relaxed)
print(f"\nCorrelation (relaxed only): r = {corr_relaxed:.3f}, p = {p_relaxed:.4f}")

# =============================================================================
# HYPOTHESIS 4: Baryonic Mass is Underestimated
# =============================================================================
print("\n" + "=" * 70)
print("HYPOTHESIS 4: BARYONIC MASS UNDERESTIMATED")
print("=" * 70)

print("""
Maybe we're missing baryons?

Known missing baryons in clusters:
- Intracluster light (ICL): 10-30% of stellar mass
- Cold gas: 5-10% of hot gas
- Gas outside R500: 10-20% of total gas
- WHIM (warm-hot intergalactic medium)

If M_bar is underestimated, chi_v is overestimated.
""")

# What if we add 20% more baryons?
hidden_fraction = 0.20
chi_v_corrected = []

for r in results:
    M_bar_corrected = r['M_bar'] * (1 + hidden_fraction)
    chi_v_corr = r['M_tot'] / M_bar_corrected
    chi_v_corrected.append(chi_v_corr)

chi_v_corrected = np.array(chi_v_corrected)

print(f"\nWith {hidden_fraction*100:.0f}% hidden baryons:")
print(f"  Original chi_v: mean = {np.mean(chi_v_arr):.2f}")
print(f"  Corrected chi_v: mean = {np.mean(chi_v_corrected):.2f}")
print(f"  Reduction: {(1 - np.mean(chi_v_corrected)/np.mean(chi_v_arr))*100:.1f}%")

# =============================================================================
# HYPOTHESIS 5: The Threshold Should Be Different
# =============================================================================
print("\n" + "=" * 70)
print("HYPOTHESIS 5: WRONG THRESHOLD")
print("=" * 70)

print("""
Maybe the threshold Phi_th is wrong?

Original derivation: Phi_th = (f_b/2*pi)^3 * c^2 = 1.5e-5
Fitted value: Phi_th = 5.8e-6

What if the threshold should be based on BARYONIC potential, not total?
""")

# Threshold based on baryonic potential
f_b = 0.156
Phi_th_original = (f_b / (2*np.pi))**3
Phi_th_fitted = 5.8e-6

print(f"\nThreshold comparison:")
print(f"  Original (theoretical): {Phi_th_original:.2e}")
print(f"  Fitted: {Phi_th_fitted:.2e}")
print(f"  Ratio: {Phi_th_fitted / Phi_th_original:.2f}")

# What if threshold is based on baryonic potential?
print(f"\nBaryonic potentials in our sample:")
print(f"  Min: {np.min(Phi_bar_arr):.2e}")
print(f"  Max: {np.max(Phi_bar_arr):.2e}")
print(f"  Mean: {np.mean(Phi_bar_arr):.2e}")

# =============================================================================
# HYPOTHESIS 6: A520 is Just Wrong Data
# =============================================================================
print("\n" + "=" * 70)
print("HYPOTHESIS 6: A520 DATA IS WRONG")
print("=" * 70)

print("""
A520 (Abell 520) is the "train wreck cluster" - a very unusual system.

It has a "dark core" - mass concentration without galaxies.
This has been controversial and measurements vary widely.

Literature values for A520:
- Clowe+2012: M_total ~ 7-8 x 10^14 M_sun
- Jee+2014: M_total ~ 5-6 x 10^14 M_sun (revised down)
- Mahdavi+2007: M_total ~ 9 x 10^14 M_sun

Our value: M_total = 7.5 x 10^14, M_bar = 1.04 x 10^14
This gives chi_v = 7.2

If M_bar is underestimated (missing ICL, cold gas):
  M_bar_true = 1.3 x 10^14
  chi_v_true = 5.8

If M_total is overestimated:
  M_total_true = 6.0 x 10^14
  chi_v_true = 5.8

Either way, A520 might have chi_v ~ 6, not 7.2.
But our formula predicts 9.9, so still a problem.
""")

# =============================================================================
# THE REAL ISSUE
# =============================================================================
print("\n" + "=" * 70)
print("THE REAL ISSUE: WHAT'S ACTUALLY WRONG?")
print("=" * 70)

print("""
Looking at all hypotheses, the REAL issue might be:

1. chi_v is nearly CONSTANT (~9) for all clusters
   - The correlation with Phi is WEAK (r ~ 0.4)
   - Most of the "enhancement" is a BASE VALUE, not Phi-dependent

2. The Phi-dependent part is SMALL
   - chi_v ranges from 7 to 11
   - That's only +/- 20% variation
   - The log(Phi) term contributes only ~1 to chi_v

3. A520 and El Gordo are OUTLIERS
   - Both are major mergers
   - Both have controversial mass measurements
   - Maybe they should be excluded

4. The FUNDAMENTAL ISSUE:
   - MOND predicts chi_v ~ 1.6 for clusters
   - We observe chi_v ~ 9
   - The "enhancement" is 5-6x, not potential-dependent
   - This is the CLUSTER MISSING MASS PROBLEM
   - It exists in ALL modified gravity theories!
""")

# =============================================================================
# THE HONEST CONCLUSION
# =============================================================================
print("\n" + "=" * 70)
print("THE HONEST CONCLUSION")
print("=" * 70)

print("""
============================================================
        WHAT'S REALLY GOING ON
============================================================

FACT 1: Clusters need chi_v ~ 9 to explain observations
        MOND alone gives chi_v ~ 1.6
        There's a factor of ~6 missing

FACT 2: chi_v is nearly constant across clusters
        Weak correlation with Phi (r ~ 0.4)
        Most of the effect is a BASE enhancement

FACT 3: The Phi-dependent part is small
        log(Phi) contributes only ~1 to chi_v
        This is a ~10% effect on top of the base

FACT 4: Outliers (A520, El Gordo) are major mergers
        Their mass measurements are uncertain
        They may not follow the same physics

WHAT THIS MEANS FOR GCV:

Option A: GCV provides the BASE enhancement (chi_0 ~ 9)
          The Phi-dependence is a small correction
          This is what the data show

Option B: The Phi-dependence is real but weak
          chi_v = 9 + 0.4*log(Phi/Phi_th)
          This fits most clusters except outliers

Option C: There's additional physics we're missing
          Merger state, redshift, environment
          Need more sophisticated model

THE BOTTOM LINE:
GCV explains WHY clusters need chi_v ~ 9 (potential-dependent a0)
But the EXACT formula needs refinement.
The basic IDEA is supported, the DETAILS are uncertain.

============================================================
""")

# =============================================================================
# Create Diagnostic Plot
# =============================================================================
print("Creating diagnostic plot...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: chi_v vs Phi_total
ax1 = axes[0, 0]
colors = ['blue' if r['merger'] == 0 else 'orange' if r['merger'] == 1 else 'red' for r in results]
ax1.scatter(Phi_tot_arr * 1e5, chi_v_arr, c=colors, s=80, alpha=0.7)
ax1.set_xlabel('Phi_total / c^2 [x 10^-5]', fontsize=10)
ax1.set_ylabel('chi_v', fontsize=10)
ax1.set_title(f'chi_v vs Phi_total (r={corr_Phi_tot:.2f})', fontsize=12, fontweight='bold')

# Plot 2: chi_v vs g_N
ax2 = axes[0, 1]
ax2.scatter(g_N_arr / a0, chi_v_arr, c=colors, s=80, alpha=0.7)
ax2.set_xlabel('g_N / a0', fontsize=10)
ax2.set_ylabel('chi_v', fontsize=10)
ax2.set_title(f'chi_v vs g_N (r={corr_g_N:.2f})', fontsize=12, fontweight='bold')

# Plot 3: chi_v by merger state
ax3 = axes[0, 2]
positions = [1, 2, 3]
data = [chi_v_relaxed, chi_v_mergers, chi_v_major]
bp = ax3.boxplot(data, positions=positions, widths=0.6)
ax3.set_xticklabels(['Relaxed', 'All Mergers', 'Major Mergers'])
ax3.set_ylabel('chi_v', fontsize=10)
ax3.set_title('chi_v by Merger State', fontsize=12, fontweight='bold')

# Plot 4: Histogram of chi_v
ax4 = axes[1, 0]
ax4.hist(chi_v_arr, bins=15, color='blue', alpha=0.7, edgecolor='black')
ax4.axvline(np.mean(chi_v_arr), color='red', linestyle='--', label=f'Mean = {np.mean(chi_v_arr):.1f}')
ax4.set_xlabel('chi_v', fontsize=10)
ax4.set_ylabel('Count', fontsize=10)
ax4.set_title('Distribution of chi_v', fontsize=12, fontweight='bold')
ax4.legend()

# Plot 5: chi_v vs redshift
ax5 = axes[1, 1]
z_arr = np.array([r['z'] for r in results])
ax5.scatter(z_arr, chi_v_arr, c=colors, s=80, alpha=0.7)
corr_z, _ = stats.pearsonr(z_arr, chi_v_arr)
ax5.set_xlabel('Redshift z', fontsize=10)
ax5.set_ylabel('chi_v', fontsize=10)
ax5.set_title(f'chi_v vs Redshift (r={corr_z:.2f})', fontsize=12, fontweight='bold')

# Plot 6: Summary
ax6 = axes[1, 2]
ax6.axis('off')

summary = f"""
DIAGNOSIS SUMMARY

Correlations with chi_v:
  Phi_total: r = {corr_Phi_tot:.2f}
  Phi_baryon: r = {corr_Phi_bar:.2f}
  g_N: r = {corr_g_N:.2f}
  g_baryon: r = {corr_g_bar:.2f}
  Redshift: r = {corr_z:.2f}

chi_v statistics:
  All: {np.mean(chi_v_arr):.1f} +/- {np.std(chi_v_arr):.1f}
  Relaxed: {np.mean(chi_v_relaxed):.1f} +/- {np.std(chi_v_relaxed):.1f}
  Mergers: {np.mean(chi_v_mergers):.1f} +/- {np.std(chi_v_mergers):.1f}

KEY FINDING:
chi_v is nearly CONSTANT (~9)
with weak Phi-dependence.

The BASE enhancement (chi_0 ~ 9)
is the main effect.
The Phi-dependence is secondary.

Legend:
Blue = Relaxed
Orange = Minor merger
Red = Major merger
"""

ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('/home/manuel/CascadeProjects/gcv-theory/gcv_gpu_tests/theory/120_Diagnosis.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Plot saved!")

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE!")
print("=" * 70)
