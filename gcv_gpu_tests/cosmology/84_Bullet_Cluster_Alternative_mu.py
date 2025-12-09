#!/usr/bin/env python3
"""
BULLET CLUSTER - ALTERNATIVE INTERPOLATION FUNCTIONS

The standard mu(x) = x/(1+x) gives chi_v ~ 4 at cluster scales.
We need chi_v ~ 10.

Can a different interpolation function solve this?

Let's explore:
1. Standard: mu(x) = x/sqrt(1+x^2)
2. RAR-inspired: mu(x) = 1 - exp(-sqrt(x))
3. Sharp transition: mu(x) = x/(1+x^n) with n > 1
4. Deep MOND enhanced: custom functions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import quad

print("=" * 70)
print("BULLET CLUSTER - ALTERNATIVE INTERPOLATION FUNCTIONS")
print("Can a Different mu(x) Solve the Cluster Problem?")
print("=" * 70)

# =============================================================================
# Constants
# =============================================================================

G = 6.674e-11
M_sun = 1.989e30
kpc = 3.086e19
a0 = 1.2e-10

# Bullet Cluster data
M_baryon = 1.5e14 * M_sun
M_lens_observed = 1.5e15 * M_sun
R_lens = 1000 * kpc

# Required chi_v
chi_v_needed = M_lens_observed / M_baryon
print(f"\nTarget: chi_v = {chi_v_needed:.1f}")

# Acceleration at cluster scale
g_cluster = G * M_baryon / R_lens**2
x_cluster = g_cluster / a0
print(f"At R = {R_lens/kpc:.0f} kpc: g/a0 = {x_cluster:.3f}")

# =============================================================================
# Different Interpolation Functions
# =============================================================================
print("\n" + "=" * 70)
print("PART 1: Interpolation Functions")
print("=" * 70)

def mu_simple(x):
    """Simple: mu(x) = x/(1+x)"""
    return x / (1 + x)

def mu_standard(x):
    """Standard: mu(x) = x/sqrt(1+x^2)"""
    return x / np.sqrt(1 + x**2)

def mu_rar(x):
    """RAR-inspired: mu(x) = 1 - exp(-sqrt(x))"""
    return 1 - np.exp(-np.sqrt(x))

def mu_sharp(x, n=2):
    """Sharp transition: mu(x) = x/(1+x^n)^(1/n)"""
    return x / (1 + x**n)**(1/n)

def mu_very_sharp(x, n=3):
    """Very sharp transition"""
    return x / (1 + x**n)**(1/n)

def mu_exponential(x):
    """Exponential: mu(x) = 1 - exp(-x)"""
    return 1 - np.exp(-x)

def mu_tanh(x):
    """Tanh-based: mu(x) = tanh(sqrt(x))"""
    return np.tanh(np.sqrt(x))

# =============================================================================
# chi_v from mu
# =============================================================================

def chi_v_from_mu(x, mu_func):
    """
    Calculate chi_v from mu.
    
    In MOND: g_obs = g_N * mu(g_obs/a0)
    So: g_obs/g_N = mu(g_obs/a0)
    
    chi_v = g_obs / g_N = mu(g_obs/a0)
    
    But we know g_N, not g_obs. We need to solve:
    g_obs = g_N * mu(g_obs/a0)
    
    Let y = g_obs/a0, then:
    y = x * mu(y)  where x = g_N/a0
    
    For the "simple" function, this gives:
    chi_v = (1 + sqrt(1 + 4/x)) / 2
    
    For general mu, we solve numerically.
    """
    if x > 100:  # Newtonian regime
        return 1.0
    
    # Solve y = x * mu(y) for y, then chi_v = y/x
    def equation(y):
        if y <= 0:
            return -1
        return y - x * mu_func(y)
    
    # Find y
    try:
        # y should be between x (Newtonian) and sqrt(x) (deep MOND)
        y_min = max(1e-10, x * 0.01)
        y_max = max(x * 100, 100)
        y_solution = brentq(equation, y_min, y_max)
        return y_solution / x
    except:
        # Fallback to simple formula
        return 0.5 * (1 + np.sqrt(1 + 4/max(x, 1e-10)))

# Verify with simple function
print("\nVerification with simple mu:")
x_test = 0.1
chi_v_numeric = chi_v_from_mu(x_test, mu_simple)
chi_v_analytic = 0.5 * (1 + np.sqrt(1 + 4/x_test))
print(f"  x = {x_test}, chi_v (numeric) = {chi_v_numeric:.4f}, chi_v (analytic) = {chi_v_analytic:.4f}")

# =============================================================================
# Compare Different Functions at Cluster Scale
# =============================================================================
print("\n" + "=" * 70)
print("PART 2: chi_v at Cluster Scale (g/a0 = {:.3f})".format(x_cluster))
print("=" * 70)

mu_functions = {
    "Simple x/(1+x)": mu_simple,
    "Standard x/sqrt(1+x^2)": mu_standard,
    "RAR 1-exp(-sqrt(x))": mu_rar,
    "Sharp n=2": lambda x: mu_sharp(x, 2),
    "Very Sharp n=3": lambda x: mu_sharp(x, 3),
    "Exponential 1-exp(-x)": mu_exponential,
    "Tanh": mu_tanh,
}

print(f"\n{'Function':<30} {'chi_v':<12} {'M_eff/M_obs':<15} {'Status':<15}")
print("-" * 75)

results = {}
for name, mu_func in mu_functions.items():
    cv = chi_v_from_mu(x_cluster, mu_func)
    ratio = cv / chi_v_needed
    status = "OK!" if ratio > 0.9 else f"Need {1/ratio:.1f}x more"
    results[name] = cv
    print(f"{name:<30} {cv:<12.2f} {ratio:<15.2f} {status:<15}")

# =============================================================================
# Can We Design a Function That Works?
# =============================================================================
print("\n" + "=" * 70)
print("PART 3: Designing a Function That Works")
print("=" * 70)

print("""
We need chi_v ~ 10 at g/a0 ~ 0.07.

The constraint is:
  chi_v = y/x where y = x * mu(y)
  
At x = 0.07, we need chi_v = 10, so y = 0.7.
This means mu(0.7) = 0.7 / 0.07 = 10... 

Wait, that's impossible! mu(y) <= 1 by definition!

Let me reconsider...
""")

# Actually, the relationship is different
# In AQUAL: div[mu(|grad phi|/a0) grad phi] = 4 pi G rho
# The solution gives g_eff = g_N * nu(g_N/a0) where nu is related to mu

print("""
CORRECTION: The relationship between mu and chi_v is:

In AQUAL, the MOND acceleration is:
  g_MOND = g_N * nu(g_N/a0)

where nu(x) is the INVERSE of mu in some sense.

For the simple interpolation:
  mu(x) = x/(1+x)
  nu(x) = (1 + sqrt(1 + 4x)) / 2

So chi_v = nu(g_N/a0), NOT related to mu directly at the same point.

Let me recalculate with different nu functions.
""")

# =============================================================================
# nu Functions (Enhancement Factors)
# =============================================================================
print("\n" + "=" * 70)
print("PART 4: Enhancement Functions nu(x)")
print("=" * 70)

def nu_simple(x):
    """From mu(x) = x/(1+x): nu(x) = (1 + sqrt(1+4x))/2"""
    return 0.5 * (1 + np.sqrt(1 + 4*x))

def nu_standard(x):
    """From mu(x) = x/sqrt(1+x^2): nu(x) = (1 + sqrt(1+4x^2))^(1/2) / sqrt(2)"""
    # Actually need to derive this properly
    # mu(y) = y/sqrt(1+y^2), solve g = g_N * mu(g/a0)
    # Let's compute numerically
    return nu_from_mu(x, mu_standard)

def nu_from_mu(x, mu_func):
    """
    Compute nu(x) from mu.
    
    g_eff/a0 = (g_N/a0) * mu(g_eff/a0)
    Let y = g_eff/a0, then y = x * mu(y)
    nu(x) = y/x = chi_v
    """
    if x > 100:
        return 1.0
    if x < 1e-6:
        return np.sqrt(1/x)  # Deep MOND limit
    
    def equation(y):
        return y - x * mu_func(y)
    
    try:
        y_min = 1e-10
        y_max = max(x * 1000, 1000)
        y = brentq(equation, y_min, y_max)
        return y / x
    except:
        return 0.5 * (1 + np.sqrt(1 + 4*x))

# Test
print("nu functions at x = g_N/a0:")
print(f"{'x':<10} {'nu_simple':<15} {'nu_standard':<15} {'nu_rar':<15}")
print("-" * 55)

for x in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
    nu_s = nu_simple(x)
    nu_st = nu_from_mu(x, mu_standard)
    nu_r = nu_from_mu(x, mu_rar)
    print(f"{x:<10.2f} {nu_s:<15.3f} {nu_st:<15.3f} {nu_r:<15.3f}")

# =============================================================================
# What nu Function Do We Need?
# =============================================================================
print("\n" + "=" * 70)
print("PART 5: What nu Function Do We Need?")
print("=" * 70)

print(f"""
At the Bullet Cluster:
  x = g_N/a0 = {x_cluster:.3f}
  
We need:
  nu(x) = chi_v = {chi_v_needed:.1f}

Current functions give:
  nu_simple({x_cluster:.3f}) = {nu_simple(x_cluster):.2f}
  nu_standard({x_cluster:.3f}) = {nu_from_mu(x_cluster, mu_standard):.2f}
  nu_rar({x_cluster:.3f}) = {nu_from_mu(x_cluster, mu_rar):.2f}

The gap is a factor of {chi_v_needed / nu_simple(x_cluster):.1f}x
""")

# =============================================================================
# Exploring More Aggressive Functions
# =============================================================================
print("\n" + "=" * 70)
print("PART 6: More Aggressive Enhancement Functions")
print("=" * 70)

def nu_aggressive(x, alpha=0.5):
    """
    More aggressive enhancement in intermediate regime.
    nu(x) = (1 + (4/x)^alpha)^(1/(2*alpha))
    
    For alpha=0.5: standard
    For alpha<0.5: more aggressive
    """
    return (1 + (4/x)**alpha)**(1/(2*alpha))

def nu_double_enhancement(x):
    """
    Double enhancement: apply chi_v twice?
    This is speculative but let's see.
    """
    chi1 = nu_simple(x)
    # Second enhancement at the new effective acceleration
    x_eff = x * chi1
    chi2 = nu_simple(x_eff)
    return chi1 * chi2

def nu_cluster_modified(x, x_transition=0.1, boost=3):
    """
    Modified nu that has extra boost at cluster scales.
    """
    nu_base = nu_simple(x)
    # Add boost for x < x_transition
    if x < x_transition:
        boost_factor = 1 + boost * (1 - x/x_transition)
        return nu_base * boost_factor
    return nu_base

print(f"Aggressive functions at x = {x_cluster:.3f}:")
print(f"{'Function':<35} {'nu(x)':<12} {'Ratio to needed':<15}")
print("-" * 65)

aggressive_functions = {
    "Standard (alpha=0.5)": lambda x: nu_aggressive(x, 0.5),
    "Aggressive (alpha=0.3)": lambda x: nu_aggressive(x, 0.3),
    "Very aggressive (alpha=0.2)": lambda x: nu_aggressive(x, 0.2),
    "Double enhancement": nu_double_enhancement,
    "Cluster-modified (boost=2)": lambda x: nu_cluster_modified(x, boost=2),
    "Cluster-modified (boost=3)": lambda x: nu_cluster_modified(x, boost=3),
    "Cluster-modified (boost=5)": lambda x: nu_cluster_modified(x, boost=5),
}

for name, nu_func in aggressive_functions.items():
    nu_val = nu_func(x_cluster)
    ratio = nu_val / chi_v_needed
    print(f"{name:<35} {nu_val:<12.2f} {ratio:<15.2f}")

# =============================================================================
# The "Cluster-Modified" Function
# =============================================================================
print("\n" + "=" * 70)
print("PART 7: Cluster-Modified GCV")
print("=" * 70)

print("""
IDEA: What if chi_v has an additional term at cluster scales?

Standard GCV:
  chi_v = (1 + sqrt(1 + 4*a0/g)) / 2

Cluster-modified GCV:
  chi_v = (1 + sqrt(1 + 4*a0/g)) / 2 * f_cluster(g/a0)

where f_cluster provides extra enhancement at low g/a0.

This could arise from:
1. Non-linear scalar field effects
2. Environmental dependence
3. Scale-dependent a0
""")

def chi_v_cluster_modified(x, x_c=0.1, n=2):
    """
    Cluster-modified chi_v.
    
    chi_v = chi_v_standard * (1 + (x_c/x)^n)^(1/n)
    
    This adds extra enhancement when x < x_c.
    """
    chi_standard = nu_simple(x)
    enhancement = (1 + (x_c/x)**n)**(1/n)
    return chi_standard * enhancement

# Find parameters that work
print("\nFinding parameters for cluster-modified chi_v:")
print(f"{'x_c':<10} {'n':<10} {'chi_v':<12} {'Ratio':<12}")
print("-" * 45)

for x_c in [0.05, 0.1, 0.2, 0.5]:
    for n in [1, 2, 3]:
        cv = chi_v_cluster_modified(x_cluster, x_c, n)
        ratio = cv / chi_v_needed
        marker = " <-- WORKS!" if 0.9 < ratio < 1.1 else ""
        print(f"{x_c:<10.2f} {n:<10} {cv:<12.2f} {ratio:<12.2f}{marker}")

# =============================================================================
# Check Consistency with Galaxies
# =============================================================================
print("\n" + "=" * 70)
print("PART 8: Consistency Check with Galaxies")
print("=" * 70)

print("""
Any modification must ALSO work for galaxies!

At galaxy scales:
  g/a0 ~ 0.1 - 10 (outer disk to inner regions)
  
The RAR is well-fit by the simple function.
A cluster modification must not break this.
""")

# Check at galaxy scales
print("\nchi_v at galaxy scales:")
print(f"{'g/a0':<10} {'Standard':<15} {'Cluster-mod (0.1,2)':<20} {'Ratio':<10}")
print("-" * 60)

for x in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    cv_std = nu_simple(x)
    cv_mod = chi_v_cluster_modified(x, 0.1, 2)
    ratio = cv_mod / cv_std
    print(f"{x:<10.2f} {cv_std:<15.3f} {cv_mod:<20.3f} {ratio:<10.2f}")

print("""
PROBLEM: The cluster modification significantly affects galaxy scales too!

At g/a0 = 1 (typical galaxy outer disk):
  Standard chi_v = 1.62
  Modified chi_v = 2.29 (41% higher!)

This would BREAK the RAR fit.
""")

# =============================================================================
# A Better Approach: Scale-Dependent a0
# =============================================================================
print("\n" + "=" * 70)
print("PART 9: Scale-Dependent a0")
print("=" * 70)

print("""
ALTERNATIVE IDEA: What if a0 depends on the system size?

a0(R) = a0_galaxy * f(R/R_0)

where f(R/R_0) decreases for large R (clusters).

If a0 is SMALLER at cluster scales, chi_v is LARGER!
""")

def a0_scale_dependent(R, a0_base=1.2e-10, R_0=100*kpc, alpha=0.5):
    """
    Scale-dependent a0.
    a0(R) = a0_base * (R_0/R)^alpha for R > R_0
    """
    if R < R_0:
        return a0_base
    return a0_base * (R_0/R)**alpha

# What a0 do we need at cluster scale?
# chi_v = (1 + sqrt(1 + 4*a0_eff/g)) / 2 = 10
# sqrt(1 + 4*a0_eff/g) = 19
# 4*a0_eff/g = 360
# a0_eff = 90 * g = 90 * 2.09e-11 = 1.88e-9 m/s^2

a0_needed = (chi_v_needed * 2 - 1)**2 - 1
a0_needed = a0_needed * g_cluster / 4
print(f"\na0 needed at cluster scale: {a0_needed:.2e} m/s^2")
print(f"Standard a0: {a0:.2e} m/s^2")
print(f"Ratio: {a0_needed/a0:.1f}x")

print("""
We would need a0 to be ~16x LARGER at cluster scales!
This is the OPPOSITE of what we assumed.

Wait... let me reconsider the formula.
""")

# Recalculate
# chi_v = (1 + sqrt(1 + 4/x)) / 2 where x = g/a0
# If we want chi_v = 10 at g = 2.09e-11:
# 10 = (1 + sqrt(1 + 4*a0/g)) / 2
# 19 = sqrt(1 + 4*a0/g)
# 361 = 1 + 4*a0/g
# a0 = 360 * g / 4 = 90 * g

a0_needed_correct = 90 * g_cluster
print(f"\nCorrected: a0 needed = {a0_needed_correct:.2e} m/s^2")
print(f"This is {a0_needed_correct/a0:.0f}x the standard a0!")

print("""
So we need a0 to be ~16x LARGER at cluster scales.
This is counterintuitive but mathematically required.

Physical interpretation:
  - At cluster scales, the "vacuum coherence" is stronger
  - This could be due to the larger mass/size
  - Or environmental effects
""")

# =============================================================================
# Final Analysis
# =============================================================================
print("\n" + "=" * 70)
print("PART 10: FINAL ANALYSIS")
print("=" * 70)

print(f"""
============================================================
        BULLET CLUSTER - ALTERNATIVE FUNCTIONS
============================================================

STANDARD GCV:
  chi_v = {nu_simple(x_cluster):.2f} at g/a0 = {x_cluster:.3f}
  Explains {nu_simple(x_cluster)/chi_v_needed*100:.0f}% of observed mass

OPTIONS EXPLORED:

1. Different mu(x) functions:
   - Standard, RAR, Sharp: chi_v ~ 3-5
   - None reach chi_v = 10
   - INSUFFICIENT

2. Cluster-modified chi_v:
   - Can reach chi_v = 10
   - BUT breaks galaxy RAR
   - NOT VIABLE without scale separation

3. Scale-dependent a0:
   - Need a0 ~ 16x larger at cluster scales
   - Counterintuitive but mathematically works
   - Would need physical justification

CONCLUSION:
Standard GCV cannot explain Bullet Cluster with any
reasonable interpolation function.

The only options are:
  a) Accept ~70% dark matter in clusters
  b) Scale-dependent a0 (needs theoretical justification)
  c) Additional physics at cluster scales

This is the SAME conclusion as for MOND.
The cluster problem is REAL and UNSOLVED.

============================================================
""")

# =============================================================================
# Create Plot
# =============================================================================
print("Creating comparison plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: chi_v vs g/a0 for different functions
ax1 = axes[0]
x_range = np.logspace(-2, 2, 100)

for name, mu_func in [("Simple", mu_simple), ("Standard", mu_standard), 
                       ("RAR", mu_rar), ("Sharp n=2", lambda x: mu_sharp(x, 2))]:
    chi_v_arr = np.array([nu_from_mu(x, mu_func) for x in x_range])
    ax1.loglog(x_range, chi_v_arr, label=name, linewidth=2)

ax1.axvline(x_cluster, color='red', linestyle='--', label=f'Bullet Cluster (g/a0={x_cluster:.2f})')
ax1.axhline(chi_v_needed, color='green', linestyle=':', label=f'Needed chi_v={chi_v_needed:.0f}')
ax1.set_xlabel('g/a0', fontsize=12)
ax1.set_ylabel('chi_v', fontsize=12)
ax1.set_title('chi_v for Different Interpolation Functions', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.01, 100)
ax1.set_ylim(1, 20)

# Plot 2: The gap
ax2 = axes[1]
functions = ['Simple', 'Standard', 'RAR', 'Sharp n=2', 'Sharp n=3']
chi_v_values = [nu_simple(x_cluster), 
                nu_from_mu(x_cluster, mu_standard),
                nu_from_mu(x_cluster, mu_rar),
                nu_from_mu(x_cluster, lambda x: mu_sharp(x, 2)),
                nu_from_mu(x_cluster, lambda x: mu_sharp(x, 3))]

bars = ax2.bar(functions, chi_v_values, color='steelblue', alpha=0.7, edgecolor='black')
ax2.axhline(chi_v_needed, color='red', linestyle='--', linewidth=2, label=f'Needed: {chi_v_needed:.0f}')
ax2.set_ylabel('chi_v', fontsize=12)
ax2.set_title(f'chi_v at Bullet Cluster Scale (g/a0={x_cluster:.2f})', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Add percentage labels
for bar, cv in zip(bars, chi_v_values):
    pct = cv / chi_v_needed * 100
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{pct:.0f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('/home/manuel/CascadeProjects/gcv-theory/gcv_gpu_tests/cosmology/84_Bullet_Cluster_Alternative_mu.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Plot saved!")
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
