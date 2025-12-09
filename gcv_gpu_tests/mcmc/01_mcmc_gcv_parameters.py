#!/usr/bin/env python3
"""
WEEK 1 - MCMC Parameter Fitting for GCV
Usa PyMC + JAX per fit robusto con GPU

Goal: Trova (a0, amp0, gamma, beta) con uncertainties rigorose
"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import corner
import json
from pathlib import Path

# Constants
G = 6.674e-11  # m^3 kg^-1 s^-2
M_sun = 1.989e30  # kg
kpc = 3.086e19  # m
pc = 3.086e16  # m
ALPHA = 2.0

# Output paths
RESULTS_DIR = Path("../results")
PLOTS_DIR = Path("../plots")
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

print("="*60)
print("GCV MCMC PARAMETER FITTING")
print("="*60)

# Load data (TODO: replace with real data)
print("\n[1/6] Loading lensing data...")
# Per ora uso dati mock - sostituiremo con real data
R_kpc = np.array([50, 100, 200, 500, 1000])
M_star_bins = [5e10, 2e11, 3e10, 1e11]  # 4 mass bins

# Mock observations (da sostituire con real data)
# Format: for each mass bin, DeltaSigma at each R
obs_data = {
    5e10: np.array([140, 95, 55, 22, 10]),
    2e11: np.array([220, 150, 85, 35, 15]),
    3e10: np.array([110, 70, 40, 16, 7]),
    1e11: np.array([180, 115, 65, 26, 11]),
}
obs_errors = {k: v * 0.25 for k, v in obs_data.items()}  # 25% errors

print(f"Loaded {len(M_star_bins)} mass bins")
print(f"  R range: {R_kpc[0]}-{R_kpc[-1]} kpc")

# GCV model function (PyMC compatible - no if statements!)
def DeltaSigma_GCV_single(M_star, R_kpc, a0, amp0, gamma, beta):
    """GCV prediction for SINGLE R value (PyMC compatible)"""
    import pytensor.tensor as pt
    
    Mb = M_star * M_sun
    v_inf = (G * Mb * a0)**(0.25)
    Lc = pt.sqrt(G * Mb / a0) / kpc
    Rt = ALPHA * Lc
    
    amp_M = amp0 * (M_star / 1e11)**gamma
    
    R_m = R_kpc * kpc
    
    # Use pt.switch instead of if/else (PyMC compatible!)
    ds_inner = v_inf**2 / (4 * G * R_m)
    ds_outer = v_inf**2 / (4 * G * (Rt*kpc)) * (Rt / R_kpc)**1.7
    ds_base = pt.switch(R_kpc < Rt, ds_inner, ds_outer)
    
    ds_base_Msun_pc2 = ds_base / (M_sun / pc**2)
    chi_v = 1 + (R_kpc / Lc)**beta
    DeltaSigma = ds_base_Msun_pc2 * amp_M * chi_v
    
    return DeltaSigma

print("\n[2/6] Setting up PyMC model...")

# Flatten data for MCMC
all_obs = []
all_errors = []
all_M_stars = []
all_Rs = []

for M_star in M_star_bins:
    for i, R in enumerate(R_kpc):
        all_obs.append(obs_data[M_star][i])
        all_errors.append(obs_errors[M_star][i])
        all_M_stars.append(M_star)
        all_Rs.append(R)

all_obs = np.array(all_obs)
all_errors = np.array(all_errors)
all_M_stars = np.array(all_M_stars)
all_Rs = np.array(all_Rs)

print(f"Total data points: {len(all_obs)}")

# PyMC model
with pm.Model() as model:
    # Priors (informative based on preliminary tests)
    a0 = pm.TruncatedNormal('a0', mu=1.72e-10, sigma=0.2e-10, lower=1e-10, upper=3e-10)
    amp0 = pm.TruncatedNormal('amp0', mu=0.93, sigma=0.2, lower=0.5, upper=2.0)
    gamma = pm.TruncatedNormal('gamma', mu=0.10, sigma=0.05, lower=0.0, upper=0.5)
    beta = pm.TruncatedNormal('beta', mu=0.90, sigma=0.1, lower=0.5, upper=1.5)
    
    # Model predictions
    predictions = []
    for i in range(len(all_obs)):
        pred = DeltaSigma_GCV_single(all_M_stars[i], all_Rs[i], a0, amp0, gamma, beta)
        predictions.append(pred)
    
    predictions = pm.math.stack(predictions)
    
    # Likelihood
    likelihood = pm.Normal('likelihood', mu=predictions, sigma=all_errors, observed=all_obs)

print("✅ Model defined")

print("\n[3/6] Running MCMC...")
print("This will take 10-30 minutes with GPU...")
print("(Be patient - this is the rigorous part!)")

with model:
    # Sample with NUTS (GPU accelerated if JAX available)
    trace = pm.sample(
        draws=5000,  # 5k draws per chain
        tune=2000,   # 2k tuning steps
        chains=4,    # 4 parallel chains
        cores=4,
        target_accept=0.95,
        return_inferencedata=True
    )

print("✅ MCMC completed!")

print("\n[4/6] Analyzing results...")

# Summary statistics
summary = az.summary(trace, hdi_prob=0.95)
print("\nParameter Summary:")
print(summary)

# Best-fit values
best_params = {
    'a0': float(trace.posterior['a0'].mean()),
    'amp0': float(trace.posterior['amp0'].mean()),
    'gamma': float(trace.posterior['gamma'].mean()),
    'beta': float(trace.posterior['beta'].mean()),
}

print("\nBest-fit Parameters:")
for key, val in best_params.items():
    print(f"  {key} = {val:.6e}")

# Save results
results = {
    'best_params': best_params,
    'summary': summary.to_dict(),
    'n_samples': len(trace.posterior.draw) * len(trace.posterior.chain),
    'convergence': {
        'r_hat_max': float(summary['r_hat'].max()),
        'ess_bulk_min': float(summary['ess_bulk'].min()),
    }
}

output_file = RESULTS_DIR / 'mcmc_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n✅ Results saved to {output_file}")

print("\n[5/6] Creating diagnostic plots...")

# Plot 1: Trace plots
fig, axes = plt.subplots(4, 2, figsize=(12, 10))
az.plot_trace(trace, var_names=['a0', 'amp0', 'gamma', 'beta'], axes=axes)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'mcmc_traces.png', dpi=300, bbox_inches='tight')
print("  ✅ Trace plots saved")

# Plot 2: Corner plot
samples = np.column_stack([
    trace.posterior['a0'].values.flatten(),
    trace.posterior['amp0'].values.flatten(),
    trace.posterior['gamma'].values.flatten(),
    trace.posterior['beta'].values.flatten(),
])

fig = corner.corner(
    samples,
    labels=['a₀', 'amp₀', 'γ', 'β'],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_fmt='.4f'
)
plt.savefig(PLOTS_DIR / 'mcmc_corner.png', dpi=300, bbox_inches='tight')
print("  ✅ Corner plot saved")

print("\n[6/6] Checking convergence...")
r_hat_max = summary['r_hat'].max()
ess_min = summary['ess_bulk'].min()

print(f"  R-hat (max): {r_hat_max:.4f} (should be < 1.01)")
print(f"  ESS (min): {ess_min:.0f} (should be > 400)")

if r_hat_max < 1.01 and ess_min > 400:
    print("\n✅ CONVERGENCE OK - Results are reliable!")
else:
    print("\n⚠️  Convergence issues - may need more samples")

print("\n" + "="*60)
print("MCMC ANALYSIS COMPLETE")
print("="*60)
print(f"\nResults: {output_file}")
print(f"Plots: {PLOTS_DIR}/mcmc_*.png")
print("\nNext: Run lensing raw data test (Week 2)")
print("="*60)
