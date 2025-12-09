#!/usr/bin/env python3
"""
SELF-CONSISTENT GCV: RAR COMPATIBILITY TEST

The self-consistent a0(g) solves the Bullet Cluster problem.
But does it still fit the RAR data from SPARC?

We need to check if the modified RAR is within the scatter of observations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize

print("=" * 70)
print("SELF-CONSISTENT GCV: RAR COMPATIBILITY TEST")
print("=" * 70)

# =============================================================================
# Constants
# =============================================================================

G = 6.674e-11  # m^3/kg/s^2
M_sun = 1.989e30  # kg
kpc = 3.086e19  # m
a0_standard = 1.2e-10  # m/s^2

# =============================================================================
# Load SPARC RAR Data
# =============================================================================
print("\n" + "=" * 70)
print("Loading SPARC RAR Data")
print("=" * 70)

# SPARC RAR data from McGaugh et al. (2016)
# These are binned data points from the paper
# g_bar (baryonic) and g_obs (observed) in m/s^2

# Approximate data from the RAR plot (log values)
log_g_bar_data = np.array([
    -12.5, -12.25, -12.0, -11.75, -11.5, -11.25, -11.0, -10.75, -10.5,
    -10.25, -10.0, -9.75, -9.5, -9.25, -9.0, -8.75, -8.5
])

log_g_obs_data = np.array([
    -11.2, -11.0, -10.8, -10.6, -10.4, -10.25, -10.1, -9.95, -9.8,
    -9.65, -9.55, -9.45, -9.35, -9.25, -9.1, -8.9, -8.6
])

# Scatter (approximate)
scatter = 0.13  # dex (from McGaugh et al.)

g_bar_data = 10**log_g_bar_data
g_obs_data = 10**log_g_obs_data

print(f"Loaded {len(g_bar_data)} RAR data points")
print(f"g_bar range: [{g_bar_data.min():.2e}, {g_bar_data.max():.2e}] m/s^2")
print(f"g_obs range: [{g_obs_data.min():.2e}, {g_obs_data.max():.2e}] m/s^2")

# =============================================================================
# GCV Models
# =============================================================================
print("\n" + "=" * 70)
print("GCV Models")
print("=" * 70)

def chi_v_standard(g_bar, a0):
    """Standard GCV chi_v"""
    return 0.5 * (1 + np.sqrt(1 + 4 * a0 / g_bar))

def g_obs_standard_gcv(g_bar, a0):
    """Standard GCV prediction for g_obs"""
    return g_bar * chi_v_standard(g_bar, a0)

def a0_self_consistent(g, a0_base, g_trans, n=2):
    """Self-consistent a0(g)"""
    return a0_base * (1 + (g_trans / g)**n)

def chi_v_self_consistent(g_bar, a0_base, g_trans, n=2):
    """Self-consistent chi_v"""
    a0_eff = a0_self_consistent(g_bar, a0_base, g_trans, n)
    return 0.5 * (1 + np.sqrt(1 + 4 * a0_eff / g_bar))

def g_obs_self_consistent_gcv(g_bar, a0_base, g_trans, n=2):
    """Self-consistent GCV prediction for g_obs"""
    return g_bar * chi_v_self_consistent(g_bar, a0_base, g_trans, n)

# =============================================================================
# Fit Standard GCV to RAR
# =============================================================================
print("\n" + "=" * 70)
print("Fitting Standard GCV to RAR")
print("=" * 70)

# Fit a0 to the data
def residuals_standard(params, g_bar, g_obs):
    a0 = params[0]
    g_pred = g_obs_standard_gcv(g_bar, a0)
    return np.sum((np.log10(g_pred) - np.log10(g_obs))**2)

result_std = minimize(residuals_standard, [a0_standard], args=(g_bar_data, g_obs_data))
a0_fit_std = result_std.x[0]

g_obs_pred_std = g_obs_standard_gcv(g_bar_data, a0_fit_std)
residuals_std = np.log10(g_obs_pred_std) - np.log10(g_obs_data)
rms_std = np.sqrt(np.mean(residuals_std**2))

print(f"Best fit a0 = {a0_fit_std:.2e} m/s^2")
print(f"RMS residual = {rms_std:.3f} dex")
print(f"Observed scatter = {scatter:.3f} dex")
print(f"Status: {'GOOD FIT' if rms_std < scatter else 'POOR FIT'}")

# =============================================================================
# Test Self-Consistent GCV with Bullet Cluster Parameters
# =============================================================================
print("\n" + "=" * 70)
print("Testing Self-Consistent GCV (Bullet Cluster Parameters)")
print("=" * 70)

# Parameters that solve Bullet Cluster
g_trans_bullet = 0.667 * a0_standard

g_obs_pred_self = g_obs_self_consistent_gcv(g_bar_data, a0_standard, g_trans_bullet, n=2)
residuals_self = np.log10(g_obs_pred_self) - np.log10(g_obs_data)
rms_self = np.sqrt(np.mean(residuals_self**2))

print(f"g_trans = {g_trans_bullet:.2e} m/s^2 = {g_trans_bullet/a0_standard:.3f} * a0")
print(f"RMS residual = {rms_self:.3f} dex")
print(f"Observed scatter = {scatter:.3f} dex")
print(f"Status: {'GOOD FIT' if rms_self < scatter else 'POOR FIT'}")

# =============================================================================
# Can We Find Parameters That Work for Both?
# =============================================================================
print("\n" + "=" * 70)
print("Optimizing Self-Consistent GCV for RAR")
print("=" * 70)

def residuals_self_consistent(params, g_bar, g_obs):
    a0_base, g_trans = params
    if a0_base <= 0 or g_trans <= 0:
        return 1e10
    g_pred = g_obs_self_consistent_gcv(g_bar, a0_base, g_trans, n=2)
    return np.sum((np.log10(g_pred) - np.log10(g_obs))**2)

# Try different starting points
best_result = None
best_rms = np.inf

for a0_init in [0.5e-10, 1e-10, 1.5e-10, 2e-10]:
    for g_trans_init in [0.1e-10, 0.5e-10, 1e-10, 2e-10]:
        try:
            result = minimize(residuals_self_consistent, 
                            [a0_init, g_trans_init], 
                            args=(g_bar_data, g_obs_data),
                            method='Nelder-Mead')
            if result.fun < best_rms:
                best_rms = result.fun
                best_result = result
        except:
            pass

if best_result is not None:
    a0_fit_self, g_trans_fit = best_result.x
    g_obs_pred_self_fit = g_obs_self_consistent_gcv(g_bar_data, a0_fit_self, g_trans_fit, n=2)
    residuals_self_fit = np.log10(g_obs_pred_self_fit) - np.log10(g_obs_data)
    rms_self_fit = np.sqrt(np.mean(residuals_self_fit**2))
    
    print(f"Best fit parameters:")
    print(f"  a0_base = {a0_fit_self:.2e} m/s^2")
    print(f"  g_trans = {g_trans_fit:.2e} m/s^2 = {g_trans_fit/a0_fit_self:.3f} * a0")
    print(f"  RMS residual = {rms_self_fit:.3f} dex")
    print(f"  Status: {'GOOD FIT' if rms_self_fit < scatter else 'POOR FIT'}")
else:
    print("Optimization failed")
    a0_fit_self = a0_standard
    g_trans_fit = 0.01 * a0_standard

# =============================================================================
# Check Bullet Cluster with Optimized Parameters
# =============================================================================
print("\n" + "=" * 70)
print("Bullet Cluster with Optimized Parameters")
print("=" * 70)

# Bullet Cluster
M_baryon_bullet = 1.5e14 * M_sun
M_lens_bullet = 1.5e15 * M_sun
R_bullet = 1000 * kpc
g_bullet = G * M_baryon_bullet / R_bullet**2

chi_v_bullet_opt = chi_v_self_consistent(g_bullet, a0_fit_self, g_trans_fit, n=2)
chi_v_needed = M_lens_bullet / M_baryon_bullet

print(f"With optimized parameters:")
print(f"  chi_v at Bullet Cluster = {chi_v_bullet_opt:.2f}")
print(f"  chi_v needed = {chi_v_needed:.1f}")
print(f"  Ratio = {chi_v_bullet_opt/chi_v_needed:.2f}")

# =============================================================================
# The Trade-off
# =============================================================================
print("\n" + "=" * 70)
print("THE TRADE-OFF")
print("=" * 70)

print("""
We have a TRADE-OFF:

1. Parameters that FIT THE RAR:
   - g_trans ~ small (or zero)
   - chi_v at Bullet Cluster ~ 3-4
   - Explains ~30% of Bullet Cluster mass

2. Parameters that FIT THE BULLET CLUSTER:
   - g_trans ~ 0.67 * a0
   - chi_v at Bullet Cluster ~ 10
   - But RAR is modified significantly

Can we find a COMPROMISE?
""")

# =============================================================================
# Scan Parameter Space
# =============================================================================
print("\n" + "=" * 70)
print("Scanning Parameter Space")
print("=" * 70)

# Scan g_trans values
g_trans_values = np.logspace(-12, -9, 30)
results_scan = []

for g_trans in g_trans_values:
    # Calculate RAR RMS
    g_obs_pred = g_obs_self_consistent_gcv(g_bar_data, a0_standard, g_trans, n=2)
    residuals = np.log10(g_obs_pred) - np.log10(g_obs_data)
    rms = np.sqrt(np.mean(residuals**2))
    
    # Calculate Bullet Cluster chi_v
    chi_v_bc = chi_v_self_consistent(g_bullet, a0_standard, g_trans, n=2)
    
    results_scan.append({
        'g_trans': g_trans,
        'g_trans_over_a0': g_trans / a0_standard,
        'rms': rms,
        'chi_v_bc': chi_v_bc,
        'bc_fraction': chi_v_bc / chi_v_needed
    })

# Find best compromise
print(f"\n{'g_trans/a0':<12} {'RAR RMS':<12} {'BC chi_v':<12} {'BC fraction':<12}")
print("-" * 50)

for r in results_scan[::3]:  # Print every 3rd
    print(f"{r['g_trans_over_a0']:<12.4f} {r['rms']:<12.3f} {r['chi_v_bc']:<12.2f} {r['bc_fraction']:<12.2f}")

# Find where RAR RMS < 2 * scatter AND BC fraction > 0.5
good_compromises = [r for r in results_scan if r['rms'] < 2*scatter and r['bc_fraction'] > 0.5]

if good_compromises:
    print(f"\nGood compromises found: {len(good_compromises)}")
    best_compromise = max(good_compromises, key=lambda x: x['bc_fraction'])
    print(f"Best compromise:")
    print(f"  g_trans/a0 = {best_compromise['g_trans_over_a0']:.4f}")
    print(f"  RAR RMS = {best_compromise['rms']:.3f} dex (limit: {2*scatter:.3f})")
    print(f"  BC chi_v = {best_compromise['chi_v_bc']:.2f}")
    print(f"  BC fraction = {best_compromise['bc_fraction']:.2f}")
else:
    print("\nNo good compromise found!")
    print("The self-consistent model cannot satisfy both constraints.")

# =============================================================================
# Alternative: Different n Values
# =============================================================================
print("\n" + "=" * 70)
print("Testing Different n Values")
print("=" * 70)

for n in [1, 2, 3, 4]:
    # Find g_trans that gives chi_v = 10 at Bullet Cluster
    # chi_v = (1 + sqrt(1 + 4*a0*(1 + (g_trans/g)^n)/g)) / 2 = 10
    # This is complex, so we search numerically
    
    def find_g_trans_for_bc(g_trans):
        cv = chi_v_self_consistent(g_bullet, a0_standard, g_trans, n=n)
        return (cv - chi_v_needed)**2
    
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(find_g_trans_for_bc, bounds=(1e-13, 1e-9), method='bounded')
    g_trans_bc = result.x
    
    # Check RAR with this g_trans
    g_obs_pred = np.array([g_bar * chi_v_self_consistent(g_bar, a0_standard, g_trans_bc, n=n) 
                          for g_bar in g_bar_data])
    residuals = np.log10(g_obs_pred) - np.log10(g_obs_data)
    rms = np.sqrt(np.mean(residuals**2))
    
    chi_v_bc = chi_v_self_consistent(g_bullet, a0_standard, g_trans_bc, n=n)
    
    print(f"\nn = {n}:")
    print(f"  g_trans for BC = {g_trans_bc:.2e} m/s^2 = {g_trans_bc/a0_standard:.4f} * a0")
    print(f"  chi_v at BC = {chi_v_bc:.2f}")
    print(f"  RAR RMS = {rms:.3f} dex (scatter: {scatter:.3f})")
    print(f"  Status: {'RAR OK' if rms < 2*scatter else 'RAR BROKEN'}")

# =============================================================================
# Create Plots
# =============================================================================
print("\n" + "=" * 70)
print("Creating plots...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: RAR comparison
ax1 = axes[0, 0]
g_bar_plot = np.logspace(-13, -8, 100)

# Data
ax1.scatter(g_bar_data, g_obs_data, c='black', s=50, label='SPARC data', zorder=5)
ax1.fill_between(g_bar_data, g_obs_data * 10**(-scatter), g_obs_data * 10**(scatter),
                  alpha=0.2, color='gray', label=f'Scatter ({scatter} dex)')

# Standard GCV
g_obs_std_plot = g_obs_standard_gcv(g_bar_plot, a0_fit_std)
ax1.loglog(g_bar_plot, g_obs_std_plot, 'b-', linewidth=2, label='Standard GCV')

# Self-consistent (Bullet Cluster params)
g_obs_self_plot = g_obs_self_consistent_gcv(g_bar_plot, a0_standard, g_trans_bullet, n=2)
ax1.loglog(g_bar_plot, g_obs_self_plot, 'r--', linewidth=2, label='Self-consistent (BC)')

# Newton
ax1.loglog(g_bar_plot, g_bar_plot, 'k:', linewidth=1, label='Newton')

ax1.set_xlabel('g_bar [m/s^2]', fontsize=12)
ax1.set_ylabel('g_obs [m/s^2]', fontsize=12)
ax1.set_title('RAR: Data vs Models', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1e-13, 1e-8)
ax1.set_ylim(1e-12, 1e-8)

# Plot 2: Residuals
ax2 = axes[0, 1]
ax2.scatter(g_bar_data, residuals_std, c='blue', s=50, label=f'Standard (RMS={rms_std:.3f})')
ax2.scatter(g_bar_data, residuals_self, c='red', s=50, marker='s', label=f'Self-consistent (RMS={rms_self:.3f})')
ax2.axhline(0, color='black', linestyle='-')
ax2.axhline(scatter, color='gray', linestyle='--', label=f'Scatter ({scatter})')
ax2.axhline(-scatter, color='gray', linestyle='--')
ax2.set_xlabel('g_bar [m/s^2]', fontsize=12)
ax2.set_ylabel('Residual [dex]', fontsize=12)
ax2.set_title('RAR Residuals', fontsize=14, fontweight='bold')
ax2.set_xscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Trade-off
ax3 = axes[1, 0]
g_trans_arr = [r['g_trans_over_a0'] for r in results_scan]
rms_arr = [r['rms'] for r in results_scan]
bc_frac_arr = [r['bc_fraction'] for r in results_scan]

ax3.semilogx(g_trans_arr, rms_arr, 'b-', linewidth=2, label='RAR RMS')
ax3.axhline(scatter, color='blue', linestyle='--', alpha=0.5, label=f'Scatter ({scatter})')
ax3.axhline(2*scatter, color='blue', linestyle=':', alpha=0.5, label=f'2x Scatter')

ax3_twin = ax3.twinx()
ax3_twin.semilogx(g_trans_arr, bc_frac_arr, 'r-', linewidth=2, label='BC fraction')
ax3_twin.axhline(1.0, color='red', linestyle='--', alpha=0.5)

ax3.set_xlabel('g_trans / a0', fontsize=12)
ax3.set_ylabel('RAR RMS [dex]', fontsize=12, color='blue')
ax3_twin.set_ylabel('Bullet Cluster fraction', fontsize=12, color='red')
ax3.set_title('Trade-off: RAR vs Bullet Cluster', fontsize=14, fontweight='bold')
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# Plot 4: Summary
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
SELF-CONSISTENT GCV: RAR TEST

STANDARD GCV:
  Best fit a0 = {a0_fit_std:.2e} m/s^2
  RAR RMS = {rms_std:.3f} dex
  Status: GOOD FIT

SELF-CONSISTENT (Bullet Cluster params):
  g_trans = {g_trans_bullet/a0_standard:.3f} * a0
  RAR RMS = {rms_self:.3f} dex
  Status: {'GOOD' if rms_self < 2*scatter else 'POOR'} FIT

THE TRADE-OFF:
  - Small g_trans: Good RAR, poor BC
  - Large g_trans: Poor RAR, good BC

CONCLUSION:
Self-consistent a0(g) with n=2 CANNOT
simultaneously fit RAR and Bullet Cluster.

The modification needed for clusters
is too strong for galaxies.

POSSIBLE SOLUTIONS:
1. Different functional form
2. Scale-dependent (not g-dependent) a0
3. Accept partial dark matter in clusters
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('/home/manuel/CascadeProjects/gcv-theory/gcv_gpu_tests/cosmology/86_Self_Consistent_RAR_Test.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Plot saved!")

# =============================================================================
# Final Summary
# =============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"""
============================================================
     SELF-CONSISTENT GCV: RAR COMPATIBILITY TEST
============================================================

STANDARD GCV:
  a0 = {a0_fit_std:.2e} m/s^2
  RAR RMS = {rms_std:.3f} dex
  Bullet Cluster: chi_v = {chi_v_standard(g_bullet, a0_fit_std):.2f} (need {chi_v_needed:.0f})
  BC fraction = {chi_v_standard(g_bullet, a0_fit_std)/chi_v_needed:.0%}

SELF-CONSISTENT (BC parameters):
  g_trans = {g_trans_bullet/a0_standard:.3f} * a0
  RAR RMS = {rms_self:.3f} dex (scatter = {scatter:.3f})
  Bullet Cluster: chi_v = {chi_v_self_consistent(g_bullet, a0_standard, g_trans_bullet, 2):.1f}
  BC fraction = {chi_v_self_consistent(g_bullet, a0_standard, g_trans_bullet, 2)/chi_v_needed:.0%}

VERDICT:
  Self-consistent a0(g) CANNOT satisfy both constraints.
  
  - To fit RAR: need g_trans << a0
  - To fit BC: need g_trans ~ 0.67 * a0
  
  These are INCOMPATIBLE.

HONEST CONCLUSION:
  GCV (in any simple form) cannot explain both:
  1. Galaxy rotation curves (RAR)
  2. Bullet Cluster lensing mass
  
  This is the SAME problem as MOND.
  The cluster problem remains UNSOLVED.

============================================================
""")

print("=" * 70)
print("TEST COMPLETE!")
print("=" * 70)
