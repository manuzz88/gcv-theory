#!/usr/bin/env python3
"""
GCV IMPLEMENTATION IN CLASS

CLASS (Cosmic Linear Anisotropy Solving System) is the standard code
for computing cosmological observables.

This script shows HOW to implement GCV in CLASS.
We provide the modified equations and the implementation strategy.

Note: Full CLASS implementation requires modifying the C code.
Here we provide the theoretical framework and a Python prototype.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.special import spherical_jn

print("=" * 70)
print("GCV IMPLEMENTATION IN CLASS")
print("=" * 70)

# =============================================================================
# Constants and Parameters
# =============================================================================

c = 3e8  # m/s
G = 6.674e-11  # m^3/kg/s^2
H0 = 2.2e-18  # s^-1 (70 km/s/Mpc)
Mpc = 3.086e22  # m

# Cosmological parameters (Planck 2018)
Omega_m = 0.315
Omega_b = 0.049
Omega_cdm = Omega_m - Omega_b
Omega_Lambda = 0.685
Omega_r = 9.2e-5
h = 0.674
n_s = 0.965
A_s = 2.1e-9

f_b = Omega_b / Omega_m

# GCV parameters
a0 = 1.2e-10  # m/s^2
Phi_th = (f_b / (2 * np.pi))**3 * c**2
alpha_gcv = 1.5
beta_gcv = 1.5

print(f"\nCosmological parameters (Planck 2018):")
print(f"  Omega_m = {Omega_m}")
print(f"  Omega_b = {Omega_b}")
print(f"  Omega_Lambda = {Omega_Lambda}")
print(f"  h = {h}")
print(f"  n_s = {n_s}")

print(f"\nGCV parameters:")
print(f"  Phi_th/c^2 = {Phi_th/c**2:.2e}")
print(f"  alpha = beta = {alpha_gcv}")

# =============================================================================
# PART 1: CLASS Structure Overview
# =============================================================================
print("\n" + "=" * 70)
print("PART 1: CLASS STRUCTURE OVERVIEW")
print("=" * 70)

print("""
CLASS solves the Einstein-Boltzmann equations:

1. BACKGROUND MODULE (background.c):
   - Solves Friedmann equations
   - Computes H(z), distances, ages
   
   GCV MODIFICATION: NONE (background unchanged)

2. THERMODYNAMICS MODULE (thermodynamics.c):
   - Recombination history
   - Visibility function
   
   GCV MODIFICATION: NONE (thermodynamics unchanged)

3. PERTURBATIONS MODULE (perturbations.c):
   - Solves perturbation equations
   - Computes transfer functions
   
   GCV MODIFICATION: Add f(Phi) factor in Poisson equation
   
4. PRIMORDIAL MODULE (primordial.c):
   - Initial conditions from inflation
   
   GCV MODIFICATION: NONE

5. NONLINEAR MODULE (nonlinear.c):
   - Halofit for nonlinear P(k)
   
   GCV MODIFICATION: Modify halo profiles for Phi > Phi_th

6. TRANSFER MODULE (transfer.c):
   - Computes C_l from transfer functions
   
   GCV MODIFICATION: NONE (uses modified perturbations)

7. SPECTRA MODULE (spectra.c):
   - Final power spectra
   
   GCV MODIFICATION: NONE
""")

# =============================================================================
# PART 2: Modified Perturbation Equations
# =============================================================================
print("\n" + "=" * 70)
print("PART 2: MODIFIED PERTURBATION EQUATIONS FOR CLASS")
print("=" * 70)

print("""
In CLASS, the perturbation equations in synchronous gauge are:

1. METRIC PERTURBATIONS:
   h' = -k^2 * eta - 2 * (a'/a) * h + 8*pi*G*a^2 * (rho + p) * theta / k^2
   eta' = (a'/a) * eta + 4*pi*G*a^2 * rho * delta / k^2

2. MATTER PERTURBATIONS:
   delta' = -theta - h'/2
   theta' = -(a'/a) * theta + k^2 * psi

where psi is related to the Newtonian potential Phi.

GCV MODIFICATION:

In the Poisson equation (relating eta to delta), we add:

   eta' = (a'/a) * eta + 4*pi*G*a^2 * rho * delta * f_GCV(Phi) / k^2

where:
   f_GCV(Phi) = 1                                      if |Phi| <= Phi_th
              = 1 + alpha * (|Phi|/Phi_th - 1)^beta    if |Phi| > Phi_th

The key insight: At linear scales, Phi << Phi_th, so f_GCV = 1.
The modification only matters at nonlinear scales (clusters).
""")

# =============================================================================
# PART 3: Python Prototype of GCV Perturbations
# =============================================================================
print("\n" + "=" * 70)
print("PART 3: PYTHON PROTOTYPE")
print("=" * 70)

def f_gcv(Phi):
    """GCV modification function"""
    x = abs(Phi) / Phi_th
    if x <= 1:
        return 1.0
    else:
        return 1.0 + alpha_gcv * (x - 1)**beta_gcv

def H_conformal(a):
    """Conformal Hubble parameter"""
    return a * H0 * np.sqrt(Omega_r/a**4 + Omega_m/a**3 + Omega_Lambda)

def perturbation_ode(y, a, k):
    """
    Simplified perturbation equations for matter + radiation
    
    y = [delta_m, theta_m, delta_r, theta_r, Phi]
    """
    delta_m, theta_m, delta_r, theta_r, Phi = y
    
    # Conformal Hubble
    H = H_conformal(a)
    
    # Densities (normalized)
    rho_m = Omega_m / a**3
    rho_r = Omega_r / a**4
    rho_tot = rho_m + rho_r
    
    # GCV factor (only matters if Phi > Phi_th)
    f = f_gcv(Phi * c**2)  # Phi is dimensionless here
    
    # Poisson equation (modified by GCV)
    # k^2 * Phi = -4*pi*G*a^2 * rho * delta * f
    # In normalized units:
    Phi_new = -1.5 * (H0/k)**2 * a * (rho_m * delta_m + rho_r * delta_r) * f
    
    # Matter equations
    ddelta_m = -theta_m - 3 * (Phi_new - Phi) / (a * H)  # simplified
    dtheta_m = -theta_m / a + k**2 * Phi / (a * H)
    
    # Radiation equations (simplified)
    ddelta_r = -4/3 * theta_r - 4 * (Phi_new - Phi) / (a * H)
    dtheta_r = k**2 * delta_r / 4 + k**2 * Phi / (a * H)
    
    # Phi evolution
    dPhi = (Phi_new - Phi) / (a * 0.01)  # relaxation
    
    return [ddelta_m, dtheta_m, ddelta_r, dtheta_r, dPhi]

print("Python prototype of GCV perturbation equations defined.")
print("This is a simplified version for demonstration.")

# =============================================================================
# PART 4: Comparison GR vs GCV
# =============================================================================
print("\n" + "=" * 70)
print("PART 4: COMPARISON GR vs GCV")
print("=" * 70)

# Solve for a range of k values
k_values = [0.001, 0.01, 0.1, 1.0]  # h/Mpc

print("\nSolving perturbation equations for different k...")
print()

# Initial conditions (matter domination)
a_init = 1e-4
a_final = 1.0

results = {}

for k in k_values:
    # Initial conditions
    delta_m_init = 1e-5
    theta_m_init = 0
    delta_r_init = 4/3 * delta_m_init
    theta_r_init = 0
    Phi_init = -1.5 * (H0/k)**2 * a_init * Omega_m/a_init**3 * delta_m_init
    
    y0 = [delta_m_init, theta_m_init, delta_r_init, theta_r_init, Phi_init]
    
    # Solve
    a_span = np.logspace(np.log10(a_init), np.log10(a_final), 500)
    
    try:
        sol = odeint(perturbation_ode, y0, a_span, args=(k,))
        results[k] = {
            'a': a_span,
            'delta_m': sol[:, 0],
            'Phi': sol[:, 4]
        }
        print(f"k = {k:.3f} h/Mpc: delta_m(a=1) = {sol[-1, 0]:.2e}")
    except Exception as e:
        print(f"k = {k:.3f} h/Mpc: Error - {e}")

# =============================================================================
# PART 5: CLASS Implementation Guide
# =============================================================================
print("\n" + "=" * 70)
print("PART 5: CLASS IMPLEMENTATION GUIDE")
print("=" * 70)

print("""
To implement GCV in CLASS, modify the following files:

1. include/common.h:
   Add GCV parameters:
   
   double phi_th;      /* GCV threshold */
   double alpha_gcv;   /* GCV exponent */
   double beta_gcv;    /* GCV power */

2. source/input.c:
   Read GCV parameters from .ini file:
   
   class_read_double("phi_th", pba->phi_th);
   class_read_double("alpha_gcv", pba->alpha_gcv);
   class_read_double("beta_gcv", pba->beta_gcv);

3. source/perturbations.c:
   Modify the Poisson equation:
   
   /* Original: */
   /* pvecmetric[ppw->index_mt_phi] = -1.5 * ... * delta_rho; */
   
   /* GCV modified: */
   double f_gcv = 1.0;
   if (fabs(phi) > pba->phi_th) {
       double x = fabs(phi) / pba->phi_th;
       f_gcv = 1.0 + pba->alpha_gcv * pow(x - 1.0, pba->beta_gcv);
   }
   pvecmetric[ppw->index_mt_phi] = -1.5 * ... * delta_rho * f_gcv;

4. Create gcv.ini:
   
   # GCV parameters
   phi_th = 1.59e-5
   alpha_gcv = 1.5
   beta_gcv = 1.5

5. Compile and run:
   
   make clean
   make
   ./class gcv.ini

EXPECTED RESULTS:
- CMB C_l: UNCHANGED (Phi << Phi_th at recombination)
- P(k) linear: UNCHANGED (Phi << Phi_th)
- P(k) nonlinear: Modified for k > 0.1 h/Mpc (clusters)
""")

# =============================================================================
# PART 6: Verification Strategy
# =============================================================================
print("\n" + "=" * 70)
print("PART 6: VERIFICATION STRATEGY")
print("=" * 70)

print("""
To verify GCV implementation in CLASS:

1. BACKGROUND CHECK:
   - Run with phi_th = infinity (GCV off)
   - Compare H(z), D_A(z), D_L(z) with standard CLASS
   - Should be IDENTICAL

2. CMB CHECK:
   - Compare C_l^TT, C_l^TE, C_l^EE
   - Should be IDENTICAL to standard CLASS
   - Difference < 0.01%

3. LINEAR P(k) CHECK:
   - Compare P(k) at z=0 for k < 0.1 h/Mpc
   - Should be IDENTICAL to standard CLASS
   - Difference < 0.1%

4. NONLINEAR P(k) CHECK:
   - Compare P(k) at z=0 for k > 0.1 h/Mpc
   - Should show GCV enhancement
   - Enhancement ~ f_gcv(Phi_cluster)

5. CLUSTER MASS FUNCTION:
   - Use modified P(k) in halo mass function
   - Compare with observed cluster counts
   - Should match without DM

6. LENSING POWER SPECTRUM:
   - Compute C_l^kk (CMB lensing)
   - Small modification at high l (cluster scales)
   - Should be consistent with Planck lensing
""")

# =============================================================================
# PART 7: Summary
# =============================================================================
print("\n" + "=" * 70)
print("PART 7: SUMMARY")
print("=" * 70)

print("""
============================================================
        GCV CLASS IMPLEMENTATION: SUMMARY
============================================================

WHAT NEEDS TO BE MODIFIED:
  - perturbations.c: Add f_gcv factor in Poisson equation
  - input.c: Read GCV parameters
  - common.h: Define GCV parameters

WHAT STAYS THE SAME:
  - background.c: Friedmann equations unchanged
  - thermodynamics.c: Recombination unchanged
  - primordial.c: Initial conditions unchanged

EXPECTED RESULTS:
  - CMB: UNCHANGED (Phi << Phi_th)
  - BAO: UNCHANGED (linear scales)
  - sigma_8: UNCHANGED (linear scales)
  - Clusters: MODIFIED (Phi > Phi_th)

COMPUTATIONAL COST:
  - Minimal: one if-statement per Poisson evaluation
  - No new ODEs to solve
  - Same runtime as standard CLASS

STATUS:
  - Theoretical framework: COMPLETE
  - Python prototype: COMPLETE
  - C implementation: TO BE DONE

The GCV modification is MINIMAL and WELL-DEFINED.
It can be implemented in CLASS with ~50 lines of code.

============================================================
""")

# =============================================================================
# PART 8: Create Plot
# =============================================================================
print("Creating plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: f_gcv function
ax1 = axes[0, 0]
Phi_range = np.linspace(0, 5 * Phi_th, 100)
f_values = [f_gcv(P) for P in Phi_range]

ax1.plot(Phi_range/Phi_th, f_values, 'b-', linewidth=2)
ax1.axvline(1, color='red', linestyle='--', label='Threshold')
ax1.axhline(1, color='gray', linestyle=':', alpha=0.5)
ax1.set_xlabel('|Phi| / Phi_th', fontsize=12)
ax1.set_ylabel('f_GCV(Phi)', fontsize=12)
ax1.set_title('GCV Modification Function', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Perturbation growth for different k
ax2 = axes[0, 1]
for k, data in results.items():
    if 'delta_m' in data:
        ax2.loglog(data['a'], np.abs(data['delta_m']), label=f'k = {k} h/Mpc')

ax2.set_xlabel('Scale factor a', fontsize=12)
ax2.set_ylabel('|delta_m|', fontsize=12)
ax2.set_title('Matter Perturbation Growth', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: CLASS modification diagram
ax3 = axes[1, 0]
ax3.axis('off')

class_diagram = """
CLASS MODIFICATION FOR GCV

┌─────────────────────────────────────────┐
│           CLASS MODULES                 │
├─────────────────────────────────────────┤
│                                         │
│  background.c      → UNCHANGED          │
│  thermodynamics.c  → UNCHANGED          │
│  primordial.c      → UNCHANGED          │
│                                         │
│  perturbations.c   → MODIFIED           │
│    └─ Poisson eq: add f_GCV(Phi)        │
│                                         │
│  nonlinear.c       → MODIFIED           │
│    └─ Halo profiles: use f_GCV          │
│                                         │
│  transfer.c        → UNCHANGED          │
│  spectra.c         → UNCHANGED          │
│                                         │
└─────────────────────────────────────────┘

MODIFICATION: ~50 lines of C code
RUNTIME: Same as standard CLASS
"""

ax3.text(0.05, 0.95, class_diagram, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Plot 4: Summary
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = """
GCV IN CLASS: IMPLEMENTATION STATUS

COMPLETED:
  [X] Theoretical framework
  [X] Modified equations derived
  [X] Python prototype
  [X] Implementation guide

TO DO:
  [ ] C code modification
  [ ] Compilation and testing
  [ ] Comparison with Planck
  [ ] Publication

KEY INSIGHT:
GCV modifies ONLY the Poisson equation,
and ONLY when Phi > Phi_th.

This means:
  - CMB unchanged
  - BAO unchanged
  - Linear P(k) unchanged
  - Only clusters affected

The modification is MINIMAL and TESTABLE.
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig('/home/manuel/CascadeProjects/gcv-theory/gcv_gpu_tests/theory/111_CLASS_Implementation.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Plot saved!")

print("\n" + "=" * 70)
print("CLASS IMPLEMENTATION GUIDE COMPLETE!")
print("=" * 70)
