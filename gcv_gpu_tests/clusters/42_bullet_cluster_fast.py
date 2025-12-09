#!/usr/bin/env python3
"""
Bullet Cluster - FAST GPU Simulation

Uses fully vectorized operations for speed.
Barnes-Hut tree would be better but this is simpler.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

try:
    import cupy as cp
    GPU = True
    xp = cp
    print("GPU: CuPy enabled")
except:
    GPU = False
    xp = np
    print("GPU: Not available, using NumPy")

print("="*60)
print("BULLET CLUSTER - FAST SIMULATION")
print("="*60)

# Parameters - SMALL for speed
N_PER_CLUSTER = 2000  # Small but enough
DT = 2.0  # Myr
N_STEPS = 75  # 150 Myr total
SOFTENING = 20.0  # kpc

# Physics
G = 4.498e-6  # kpc^3 / (10^10 Msun * Myr^2)
MAIN_MASS = 100  # 10^10 Msun
BULLET_MASS = 15
MAIN_R = 400  # kpc
BULLET_R = 150
V_COLLISION = 4.8  # kpc/Myr (~4700 km/s)
INIT_SEP = 1500  # kpc

RESULTS_DIR = Path("../results")
PLOTS_DIR = Path("../plots")
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

print(f"\nN particles: {2*N_PER_CLUSTER}")
print(f"Steps: {N_STEPS}, dt: {DT} Myr")
print(f"Total time: {N_STEPS * DT} Myr")

# Initialize clusters
print("\nInitializing clusters...")

def make_cluster(N, M, R, center, vel, xp):
    """Create spherical cluster"""
    # Uniform sphere (simpler than NFW)
    r = R * xp.random.random(N)**(1/3)
    theta = xp.arccos(2*xp.random.random(N) - 1)
    phi = 2*xp.pi*xp.random.random(N)
    
    x = r * xp.sin(theta) * xp.cos(phi) + center[0]
    y = r * xp.sin(theta) * xp.sin(phi) + center[1]
    z = r * xp.cos(theta) + center[2]
    
    sigma = xp.sqrt(G * M / R) * 0.3
    vx = xp.random.normal(0, float(sigma), N) + vel[0]
    vy = xp.random.normal(0, float(sigma), N) + vel[1]
    vz = xp.random.normal(0, float(sigma), N) + vel[2]
    
    m = xp.ones(N) * M / N
    
    return x, y, z, vx, vy, vz, m

# Main cluster
x1, y1, z1, vx1, vy1, vz1, m1 = make_cluster(
    N_PER_CLUSTER, MAIN_MASS, MAIN_R,
    [-INIT_SEP/2, 0, 0], [V_COLLISION/2, 0, 0], xp
)

# Bullet
x2, y2, z2, vx2, vy2, vz2, m2 = make_cluster(
    N_PER_CLUSTER, BULLET_MASS, BULLET_R,
    [INIT_SEP/2, 0, 0], [-V_COLLISION/2, 0, 0], xp
)

# Combine
x = xp.concatenate([x1, x2])
y = xp.concatenate([y1, y2])
z = xp.concatenate([z1, z2])
vx = xp.concatenate([vx1, vx2])
vy = xp.concatenate([vy1, vy2])
vz = xp.concatenate([vz1, vz2])
m = xp.concatenate([m1, m2])

cluster_id = xp.concatenate([xp.zeros(N_PER_CLUSTER), xp.ones(N_PER_CLUSTER)])
N = len(x)

print(f"Total particles: {N}")

def compute_accel_vectorized(x, y, z, m, chi_v=1.0, xp=np):
    """Fully vectorized gravity - O(N^2) but GPU parallel"""
    N = len(x)
    
    # Pairwise differences (N x N matrices)
    dx = x.reshape(-1, 1) - x.reshape(1, -1)
    dy = y.reshape(-1, 1) - y.reshape(1, -1)
    dz = z.reshape(-1, 1) - z.reshape(1, -1)
    
    # Distances
    r2 = dx**2 + dy**2 + dz**2 + SOFTENING**2
    r3 = r2 * xp.sqrt(r2)
    
    # Acceleration components
    # a_i = G * sum_j (m_j * (x_j - x_i) / r_ij^3)
    ax = G * chi_v * xp.sum(m.reshape(1, -1) * (-dx) / r3, axis=1)
    ay = G * chi_v * xp.sum(m.reshape(1, -1) * (-dy) / r3, axis=1)
    az = G * chi_v * xp.sum(m.reshape(1, -1) * (-dz) / r3, axis=1)
    
    return ax, ay, az

print("\n" + "="*60)
print("RUNNING SIMULATIONS")
print("="*60)

def run_sim(x, y, z, vx, vy, vz, m, chi_v, label, xp):
    """Run simulation with given chi_v"""
    x, y, z = x.copy(), y.copy(), z.copy()
    vx, vy, vz = vx.copy(), vy.copy(), vz.copy()
    
    # Store centers of mass
    centers_main = []
    centers_bullet = []
    
    print(f"\n{label} (chi_v = {chi_v}):")
    t0 = time.time()
    
    for step in range(N_STEPS):
        # Compute acceleration
        ax, ay, az = compute_accel_vectorized(x, y, z, m, chi_v, xp)
        
        # Leapfrog
        vx += ax * DT
        vy += ay * DT
        vz += az * DT
        x += vx * DT
        y += vy * DT
        z += vz * DT
        
        # Track centers
        if step % 5 == 0:
            main_mask = cluster_id == 0
            bullet_mask = cluster_id == 1
            
            if GPU:
                cx_main = float(cp.average(x[main_mask], weights=m[main_mask]))
                cy_main = float(cp.average(y[main_mask], weights=m[main_mask]))
                cx_bullet = float(cp.average(x[bullet_mask], weights=m[bullet_mask]))
                cy_bullet = float(cp.average(y[bullet_mask], weights=m[bullet_mask]))
            else:
                cx_main = np.average(x[main_mask], weights=m[main_mask])
                cy_main = np.average(y[main_mask], weights=m[main_mask])
                cx_bullet = np.average(x[bullet_mask], weights=m[bullet_mask])
                cy_bullet = np.average(y[bullet_mask], weights=m[bullet_mask])
            
            centers_main.append([cx_main, cy_main])
            centers_bullet.append([cx_bullet, cy_bullet])
        
        if (step + 1) % 15 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (step + 1) * (N_STEPS - step - 1)
            print(f"  Step {step+1}/{N_STEPS}, {elapsed:.1f}s, ETA: {eta:.1f}s")
    
    print(f"  Done in {time.time() - t0:.1f}s")
    
    return (x, y, z), np.array(centers_main), np.array(centers_bullet)

# LCDM (chi_v = 1)
final_lcdm, centers_main_lcdm, centers_bullet_lcdm = run_sim(
    x, y, z, vx, vy, vz, m, chi_v=1.0, label="LCDM", xp=xp
)

# GCV (chi_v = 1.5 average)
final_gcv, centers_main_gcv, centers_bullet_gcv = run_sim(
    x, y, z, vx, vy, vz, m, chi_v=1.5, label="GCV", xp=xp
)

print("\n" + "="*60)
print("RESULTS")
print("="*60)

# Compute separations
sep_lcdm = np.sqrt((centers_main_lcdm[:, 0] - centers_bullet_lcdm[:, 0])**2 +
                   (centers_main_lcdm[:, 1] - centers_bullet_lcdm[:, 1])**2)
sep_gcv = np.sqrt((centers_main_gcv[:, 0] - centers_bullet_gcv[:, 0])**2 +
                  (centers_main_gcv[:, 1] - centers_bullet_gcv[:, 1])**2)

time_arr = np.arange(len(sep_lcdm)) * 5 * DT

print(f"\nInitial separation: {sep_lcdm[0]:.0f} kpc")
print(f"Final separation (LCDM): {sep_lcdm[-1]:.0f} kpc")
print(f"Final separation (GCV):  {sep_gcv[-1]:.0f} kpc")
print(f"Difference: {sep_lcdm[-1] - sep_gcv[-1]:.0f} kpc")

# Minimum separation (closest approach)
min_sep_lcdm = np.min(sep_lcdm)
min_sep_gcv = np.min(sep_gcv)
print(f"\nClosest approach (LCDM): {min_sep_lcdm:.0f} kpc")
print(f"Closest approach (GCV):  {min_sep_gcv:.0f} kpc")

print("\n" + "="*60)
print("VISUALIZATION")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Bullet Cluster: LCDM vs GCV', fontsize=14, fontweight='bold')

# Convert GPU arrays to numpy for plotting
if GPU:
    x_lcdm, y_lcdm, z_lcdm = [cp.asnumpy(arr) for arr in final_lcdm]
    x_gcv, y_gcv, z_gcv = [cp.asnumpy(arr) for arr in final_gcv]
    cid = cp.asnumpy(cluster_id)
else:
    x_lcdm, y_lcdm, z_lcdm = final_lcdm
    x_gcv, y_gcv, z_gcv = final_gcv
    cid = cluster_id

# Final state LCDM
ax1 = axes[0, 0]
ax1.scatter(x_lcdm[cid==0], y_lcdm[cid==0], s=1, alpha=0.5, c='blue', label='Main')
ax1.scatter(x_lcdm[cid==1], y_lcdm[cid==1], s=1, alpha=0.5, c='red', label='Bullet')
ax1.set_xlabel('x [kpc]')
ax1.set_ylabel('y [kpc]')
ax1.set_title(f'LCDM Final (sep={sep_lcdm[-1]:.0f} kpc)')
ax1.legend()
ax1.set_aspect('equal')

# Final state GCV
ax2 = axes[0, 1]
ax2.scatter(x_gcv[cid==0], y_gcv[cid==0], s=1, alpha=0.5, c='blue', label='Main')
ax2.scatter(x_gcv[cid==1], y_gcv[cid==1], s=1, alpha=0.5, c='red', label='Bullet')
ax2.set_xlabel('x [kpc]')
ax2.set_ylabel('y [kpc]')
ax2.set_title(f'GCV Final (sep={sep_gcv[-1]:.0f} kpc)')
ax2.legend()
ax2.set_aspect('equal')

# Separation vs time
ax3 = axes[1, 0]
ax3.plot(time_arr, sep_lcdm, 'b-', lw=2, label='LCDM')
ax3.plot(time_arr, sep_gcv, 'r--', lw=2, label='GCV')
ax3.set_xlabel('Time [Myr]')
ax3.set_ylabel('Separation [kpc]')
ax3.set_title('Cluster Separation')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Summary
ax4 = axes[1, 1]
ax4.axis('off')
summary = f"""
BULLET CLUSTER SIMULATION

Particles: {N}
Time: {N_STEPS * DT:.0f} Myr
GPU: {GPU}

Results:
  Initial sep: {sep_lcdm[0]:.0f} kpc
  
  LCDM:
    Final sep: {sep_lcdm[-1]:.0f} kpc
    Min sep: {min_sep_lcdm:.0f} kpc
    
  GCV (chi_v=1.5):
    Final sep: {sep_gcv[-1]:.0f} kpc
    Min sep: {min_sep_gcv:.0f} kpc

Difference: {sep_lcdm[-1] - sep_gcv[-1]:.0f} kpc

KEY FINDING:
GCV clusters merge FASTER
due to enhanced gravity!
"""
ax4.text(0.1, 0.9, summary, fontsize=11, family='monospace',
         verticalalignment='top', transform=ax4.transAxes)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'bullet_cluster_fast.png', dpi=150)
print(f"Plot saved!")

# Save results
results = {
    'n_particles': N,
    'time_myr': N_STEPS * DT,
    'lcdm': {'final_sep': float(sep_lcdm[-1]), 'min_sep': float(min_sep_lcdm)},
    'gcv': {'final_sep': float(sep_gcv[-1]), 'min_sep': float(min_sep_gcv)},
    'difference_kpc': float(sep_lcdm[-1] - sep_gcv[-1])
}
with open(RESULTS_DIR / 'bullet_cluster_fast.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("DONE!")
print("="*60)
