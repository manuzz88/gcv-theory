#!/usr/bin/env python3
"""
Black Hole Shadow Test - EHT M87*

The Event Horizon Telescope imaged M87* black hole shadow.
The shadow size depends on the black hole mass and gravity theory.

GCV prediction:
- G_eff = G * chi_v near black hole
- Shadow size R_sh ~ G_eff * M / c^2
- Could be slightly larger than GR prediction!

Data: EHT 2019 results for M87*
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

print("="*70)
print("BLACK HOLE SHADOW TEST - M87*")
print("="*70)

# Physical constants
c = 299792458  # m/s
G = 6.674e-11  # m^3/(kg*s^2)
Msun = 1.989e30  # kg
pc = 3.086e16  # m
Mpc = 3.086e22  # m
arcsec = np.pi / (180 * 3600)  # radians

# M87* parameters from EHT
M87_data = {
    'name': 'M87*',
    'distance_Mpc': 16.8,
    'distance_err_Mpc': 0.8,
    'mass_Msun': 6.5e9,
    'mass_err_Msun': 0.7e9,
    'shadow_diameter_uas': 42.0,  # microarcseconds
    'shadow_err_uas': 3.0,
}

# Convert to SI
D = M87_data['distance_Mpc'] * Mpc
M = M87_data['mass_Msun'] * Msun
theta_obs = M87_data['shadow_diameter_uas'] * 1e-6 * arcsec  # radians

print(f"\nM87* Black Hole:")
print(f"  Distance: {M87_data['distance_Mpc']} +/- {M87_data['distance_err_Mpc']} Mpc")
print(f"  Mass: {M87_data['mass_Msun']:.1e} +/- {M87_data['mass_err_Msun']:.1e} Msun")
print(f"  Shadow diameter: {M87_data['shadow_diameter_uas']} +/- {M87_data['shadow_err_uas']} uas")

RESULTS_DIR = Path("../results")
PLOTS_DIR = Path("../plots")
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

print("\n" + "="*70)
print("STEP 1: GR PREDICTION")
print("="*70)

# In GR, the shadow of a Schwarzschild black hole has radius:
# R_shadow = sqrt(27) * G * M / c^2 = sqrt(27) * R_s / 2
# where R_s = 2*G*M/c^2 is the Schwarzschild radius

R_s = 2 * G * M / c**2
R_shadow_GR = np.sqrt(27) * G * M / c**2

# Angular size
theta_GR = 2 * R_shadow_GR / D  # diameter in radians
theta_GR_uas = theta_GR / arcsec * 1e6  # microarcseconds

print(f"\nGR Prediction:")
print(f"  Schwarzschild radius: R_s = {R_s:.2e} m = {R_s/pc:.2f} pc")
print(f"  Shadow radius: R_sh = {R_shadow_GR:.2e} m")
print(f"  Shadow diameter: {theta_GR_uas:.1f} uas")

print("\n" + "="*70)
print("STEP 2: GCV PREDICTION")
print("="*70)

# GCV parameters
a0 = 1.80e-10  # m/s^2
z0 = 10.0
alpha_z = 2.0

# For M87* at z ~ 0.004 (very nearby)
z_M87 = 0.00428

def gcv_chi_v(M_kg, r_m, z=0):
    """GCV chi_v for a given mass and radius"""
    # Coherence length
    L_c = np.sqrt(G * M_kg / a0)
    
    # Redshift factor
    f_z = 1.0 / (1 + z / z0)**alpha_z
    
    # For black holes, we're at r ~ R_s (very close)
    # chi_v should be evaluated at the photon sphere r = 1.5 * R_s
    
    # Mass factor
    M_crit = 1e10 * Msun
    f_M = 1.0 / (1 + (M_crit / M_kg)**0.5)
    
    # Radial factor
    if r_m < L_c:
        f_r = (r_m / L_c)**0.5
    else:
        f_r = 1.0
    
    # chi_v
    chi_v = 1 + 0.03 * f_z * f_M * f_r
    
    return chi_v, L_c

# Photon sphere radius
r_photon = 1.5 * R_s

chi_v_M87, L_c_M87 = gcv_chi_v(M, r_photon, z_M87)

print(f"\nGCV at M87*:")
print(f"  Photon sphere: r = {r_photon:.2e} m")
print(f"  Coherence length: L_c = {L_c_M87:.2e} m")
print(f"  r/L_c = {r_photon/L_c_M87:.2e}")
print(f"  chi_v = {chi_v_M87:.6f}")

# GCV shadow size
# In GCV, G_eff = G * chi_v
# Shadow radius scales with G_eff
R_shadow_GCV = np.sqrt(27) * G * chi_v_M87 * M / c**2
theta_GCV = 2 * R_shadow_GCV / D
theta_GCV_uas = theta_GCV / arcsec * 1e6

print(f"\nGCV Prediction:")
print(f"  G_eff/G = chi_v = {chi_v_M87:.6f}")
print(f"  Shadow radius: R_sh = {R_shadow_GCV:.2e} m")
print(f"  Shadow diameter: {theta_GCV_uas:.1f} uas")
print(f"  Difference from GR: {(theta_GCV_uas/theta_GR_uas - 1)*100:.3f}%")

print("\n" + "="*70)
print("STEP 3: COMPARISON WITH EHT DATA")
print("="*70)

theta_obs_uas = M87_data['shadow_diameter_uas']
theta_err_uas = M87_data['shadow_err_uas']

print(f"\nComparison:")
print(f"  Observed:  {theta_obs_uas:.1f} +/- {theta_err_uas:.1f} uas")
print(f"  GR:        {theta_GR_uas:.1f} uas")
print(f"  GCV:       {theta_GCV_uas:.1f} uas")

# Chi-square
chi2_GR = ((theta_obs_uas - theta_GR_uas) / theta_err_uas)**2
chi2_GCV = ((theta_obs_uas - theta_GCV_uas) / theta_err_uas)**2

print(f"\nChi-square:")
print(f"  GR:  chi2 = {chi2_GR:.2f}")
print(f"  GCV: chi2 = {chi2_GCV:.2f}")
print(f"  Delta chi2 = {chi2_GCV - chi2_GR:+.2f}")

# Sigma deviation
sigma_GR = abs(theta_obs_uas - theta_GR_uas) / theta_err_uas
sigma_GCV = abs(theta_obs_uas - theta_GCV_uas) / theta_err_uas

print(f"\nDeviation from observation:")
print(f"  GR:  {sigma_GR:.2f} sigma")
print(f"  GCV: {sigma_GCV:.2f} sigma")

print("\n" + "="*70)
print("STEP 4: MASS INFERENCE")
print("="*70)

print("""
The EHT team inferred M87* mass from the shadow size assuming GR.
If GCV is correct, the TRUE mass would be different!

From shadow size: theta = 2 * sqrt(27) * G_eff * M / (c^2 * D)

GR:  M_GR = theta * c^2 * D / (2 * sqrt(27) * G)
GCV: M_GCV = theta * c^2 * D / (2 * sqrt(27) * G * chi_v)
     M_GCV = M_GR / chi_v
""")

M_inferred_GR = theta_obs * c**2 * D / (2 * np.sqrt(27) * G)
M_inferred_GCV = M_inferred_GR / chi_v_M87

print(f"Mass inferred from shadow:")
print(f"  GR assumption:  M = {M_inferred_GR/Msun:.2e} Msun")
print(f"  GCV assumption: M = {M_inferred_GCV/Msun:.2e} Msun")
print(f"  Difference: {(1 - M_inferred_GCV/M_inferred_GR)*100:.3f}%")

print("\n" + "="*70)
print("STEP 5: KERR BLACK HOLE (SPINNING)")
print("="*70)

print("""
M87* is likely spinning (a/M ~ 0.9).
For a Kerr black hole, the shadow is slightly asymmetric.

The shadow size for Kerr depends on spin and viewing angle.
For high spin viewed edge-on:
  R_shadow ~ 4.5-5.2 * G*M/c^2 (vs 5.2 for Schwarzschild)

GCV modification still applies: G -> G * chi_v
""")

# Kerr shadow (approximate for a/M = 0.9)
a_spin = 0.9  # dimensionless spin
R_shadow_Kerr = 5.0 * G * M / c**2  # approximate
theta_Kerr_uas = 2 * R_shadow_Kerr / D / arcsec * 1e6

R_shadow_Kerr_GCV = 5.0 * G * chi_v_M87 * M / c**2
theta_Kerr_GCV_uas = 2 * R_shadow_Kerr_GCV / D / arcsec * 1e6

print(f"\nKerr black hole (a/M = {a_spin}):")
print(f"  GR shadow:  {theta_Kerr_uas:.1f} uas")
print(f"  GCV shadow: {theta_Kerr_GCV_uas:.1f} uas")

print("\n" + "="*70)
print("STEP 6: SAGITTARIUS A* PREDICTION")
print("="*70)

# Sgr A* parameters
SgrA_data = {
    'name': 'Sgr A*',
    'distance_kpc': 8.178,
    'mass_Msun': 4.15e6,
    'shadow_diameter_uas': 51.8,  # EHT 2022
    'shadow_err_uas': 2.3,
}

D_SgrA = SgrA_data['distance_kpc'] * 1000 * pc
M_SgrA = SgrA_data['mass_Msun'] * Msun

# GR prediction
R_shadow_SgrA_GR = np.sqrt(27) * G * M_SgrA / c**2
theta_SgrA_GR_uas = 2 * R_shadow_SgrA_GR / D_SgrA / arcsec * 1e6

# GCV prediction
chi_v_SgrA, L_c_SgrA = gcv_chi_v(M_SgrA, 1.5 * 2 * G * M_SgrA / c**2, z=0)
R_shadow_SgrA_GCV = np.sqrt(27) * G * chi_v_SgrA * M_SgrA / c**2
theta_SgrA_GCV_uas = 2 * R_shadow_SgrA_GCV / D_SgrA / arcsec * 1e6

print(f"\nSgr A* (Milky Way center):")
print(f"  Mass: {SgrA_data['mass_Msun']:.2e} Msun")
print(f"  Distance: {SgrA_data['distance_kpc']} kpc")
print(f"  chi_v = {chi_v_SgrA:.6f}")
print(f"\n  Observed: {SgrA_data['shadow_diameter_uas']} +/- {SgrA_data['shadow_err_uas']} uas")
print(f"  GR:       {theta_SgrA_GR_uas:.1f} uas")
print(f"  GCV:      {theta_SgrA_GCV_uas:.1f} uas")

chi2_SgrA_GR = ((SgrA_data['shadow_diameter_uas'] - theta_SgrA_GR_uas) / SgrA_data['shadow_err_uas'])**2
chi2_SgrA_GCV = ((SgrA_data['shadow_diameter_uas'] - theta_SgrA_GCV_uas) / SgrA_data['shadow_err_uas'])**2

print(f"\n  Chi2 GR:  {chi2_SgrA_GR:.2f}")
print(f"  Chi2 GCV: {chi2_SgrA_GCV:.2f}")

print("\n" + "="*70)
print("STEP 7: VERDICT")
print("="*70)

# Combined chi2
total_chi2_GR = chi2_GR + chi2_SgrA_GR
total_chi2_GCV = chi2_GCV + chi2_SgrA_GCV
delta_chi2 = total_chi2_GCV - total_chi2_GR

print(f"\nCombined results (M87* + Sgr A*):")
print(f"  Total chi2 GR:  {total_chi2_GR:.2f}")
print(f"  Total chi2 GCV: {total_chi2_GCV:.2f}")
print(f"  Delta chi2: {delta_chi2:+.2f}")

if abs(delta_chi2) < 1:
    verdict = "EQUIVALENT"
elif delta_chi2 < 0:
    verdict = "GCV_BETTER"
else:
    verdict = "GR_BETTER"

print(f"\nVERDICT: {verdict}")

print(f"""
INTERPRETATION:

GCV chi_v at black hole photon sphere is very close to 1!
- M87*: chi_v = {chi_v_M87:.6f} (modification: {(chi_v_M87-1)*100:.4f}%)
- Sgr A*: chi_v = {chi_v_SgrA:.6f} (modification: {(chi_v_SgrA-1)*100:.4f}%)

Why?
- Black holes are VERY compact: r << L_c
- GCV chi_v is suppressed at small r
- This is CONSISTENT with GCV theory!

GCV predicts:
- Shadow size ~ GR (chi_v ~ 1 near BH)
- No conflict with EHT observations!
""")

print("\n" + "="*70)
print("STEP 8: SAVE RESULTS")
print("="*70)

results = {
    'test': 'Black Hole Shadow',
    'M87': {
        'observed_uas': M87_data['shadow_diameter_uas'],
        'GR_prediction_uas': float(theta_GR_uas),
        'GCV_prediction_uas': float(theta_GCV_uas),
        'chi_v': float(chi_v_M87),
        'chi2_GR': float(chi2_GR),
        'chi2_GCV': float(chi2_GCV)
    },
    'SgrA': {
        'observed_uas': SgrA_data['shadow_diameter_uas'],
        'GR_prediction_uas': float(theta_SgrA_GR_uas),
        'GCV_prediction_uas': float(theta_SgrA_GCV_uas),
        'chi_v': float(chi_v_SgrA),
        'chi2_GR': float(chi2_SgrA_GR),
        'chi2_GCV': float(chi2_SgrA_GCV)
    },
    'combined': {
        'total_chi2_GR': float(total_chi2_GR),
        'total_chi2_GCV': float(total_chi2_GCV),
        'delta_chi2': float(delta_chi2)
    },
    'verdict': verdict
}

output_file = RESULTS_DIR / 'black_hole_shadow.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved: {output_file}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Black Hole Shadow Test: GCV vs GR', fontsize=14, fontweight='bold')

# Plot 1: M87* comparison
ax1 = axes[0, 0]
models = ['Observed', 'GR', 'GCV']
values = [theta_obs_uas, theta_GR_uas, theta_GCV_uas]
errors = [theta_err_uas, 0, 0]
colors = ['black', 'blue', 'red']
ax1.bar(models, values, yerr=errors, color=colors, alpha=0.7, capsize=5)
ax1.set_ylabel('Shadow diameter [uas]')
ax1.set_title('M87* Shadow Size')
ax1.grid(True, alpha=0.3)

# Plot 2: Sgr A* comparison
ax2 = axes[0, 1]
values_SgrA = [SgrA_data['shadow_diameter_uas'], theta_SgrA_GR_uas, theta_SgrA_GCV_uas]
errors_SgrA = [SgrA_data['shadow_err_uas'], 0, 0]
ax2.bar(models, values_SgrA, yerr=errors_SgrA, color=colors, alpha=0.7, capsize=5)
ax2.set_ylabel('Shadow diameter [uas]')
ax2.set_title('Sgr A* Shadow Size')
ax2.grid(True, alpha=0.3)

# Plot 3: chi_v vs r/R_s
ax3 = axes[1, 0]
r_over_Rs = np.logspace(-1, 3, 100)
r_m = r_over_Rs * R_s
chi_v_profile = [gcv_chi_v(M, r, z_M87)[0] for r in r_m]
ax3.semilogx(r_over_Rs, chi_v_profile, 'b-', lw=2)
ax3.axvline(1.5, color='red', linestyle='--', label='Photon sphere')
ax3.axhline(1, color='gray', linestyle='-', alpha=0.5)
ax3.set_xlabel('r / R_s')
ax3.set_ylabel('chi_v')
ax3.set_title('GCV chi_v Profile (M87*)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Summary
ax4 = axes[1, 1]
ax4.axis('off')
summary = f"""
BLACK HOLE SHADOW TEST

M87* (EHT 2019):
  Observed: {theta_obs_uas:.1f} +/- {theta_err_uas:.1f} uas
  GR:       {theta_GR_uas:.1f} uas
  GCV:      {theta_GCV_uas:.1f} uas
  chi_v = {chi_v_M87:.6f}

Sgr A* (EHT 2022):
  Observed: {SgrA_data['shadow_diameter_uas']:.1f} +/- {SgrA_data['shadow_err_uas']:.1f} uas
  GR:       {theta_SgrA_GR_uas:.1f} uas
  GCV:      {theta_SgrA_GCV_uas:.1f} uas
  chi_v = {chi_v_SgrA:.6f}

Combined chi2:
  GR:  {total_chi2_GR:.2f}
  GCV: {total_chi2_GCV:.2f}
  Delta: {delta_chi2:+.2f}

VERDICT: {verdict}

KEY: chi_v ~ 1 near black holes!
GCV is suppressed at small scales.
"""
ax4.text(0.05, 0.95, summary, fontsize=10, family='monospace',
         verticalalignment='top', transform=ax4.transAxes)

plt.tight_layout()
plot_file = PLOTS_DIR / 'black_hole_shadow.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Plot saved: {plot_file}")

print("\n" + "="*70)
print("BLACK HOLE SHADOW TEST COMPLETE!")
print("="*70)
