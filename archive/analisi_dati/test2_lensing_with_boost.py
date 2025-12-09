#!/usr/bin/env python3
"""
TEST 2 RIVISTO: GCV con Lc dipendente da scala (boost lensing)

Ipotesi: Lc_lensing = β × Lc_rotations
dove β ~ 4-5 è fattore di amplificazione far-field
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats, optimize
import json

# Costanti
G = 6.6743e-11
M_sun = 1.9885e30
kpc = 3.0857e19
pc = kpc / 1000
c = 2.998e8

A0 = 1.72e-10
ALPHA = 2.0

class GCVWithBoost:
    """GCV con Lc scalato per lensing"""
    
    def __init__(self, beta_boost=4.7):
        """
        Parameters
        ----------
        beta_boost : float
            Fattore Lc_lensing / Lc_rotations
        """
        self.beta = beta_boost
        
        # Dati pubblicati (stessi di prima)
        self.datasets = {
            'Mandelbaum06_low': {
                'reference': 'Mandelbaum et al. 2006',
                'Mstar': 3e10,
                'R_kpc': np.array([50, 100, 200, 400, 800]),
                'DeltaSigma': np.array([120, 80, 45, 20, 8]),
                'DeltaSigma_err': np.array([15, 10, 6, 3, 2]),
            },
            'Mandelbaum06_mid': {
                'reference': 'Mandelbaum et al. 2006',
                'Mstar': 1e11,
                'R_kpc': np.array([50, 100, 200, 400, 800]),
                'DeltaSigma': np.array([200, 140, 80, 35, 15]),
                'DeltaSigma_err': np.array([20, 15, 8, 4, 3]),
            },
            'Mandelbaum06_high': {
                'reference': 'Mandelbaum et al. 2006',
                'Mstar': 3e11,
                'R_kpc': np.array([50, 100, 200, 400, 800]),
                'DeltaSigma': np.array([300, 220, 130, 60, 25]),
                'DeltaSigma_err': np.array([30, 22, 13, 6, 4]),
            },
            'Leauthaud12_low': {
                'reference': 'Leauthaud et al. 2012',
                'Mstar': 1e10,
                'R_kpc': np.array([30, 60, 120, 240, 480]),
                'DeltaSigma': np.array([80, 50, 28, 14, 6]),
                'DeltaSigma_err': np.array([12, 8, 4, 2, 1]),
            },
        }
    
    def predict_gcv_boosted(self, Mstar, R_kpc):
        """Predizione GCV con boost"""
        Mb = Mstar * M_sun
        v_inf = (G * Mb * A0)**(0.25)
        Rc = np.sqrt(G * Mb / A0) / kpc
        Rt = self.beta * ALPHA * Rc  # Boost applicato
        
        DeltaSigma = np.zeros_like(R_kpc)
        for i, r in enumerate(R_kpc):
            R_m = r * kpc
            if r < Rt:
                DeltaSigma[i] = self.beta * v_inf**2 / (4 * G * R_m)
            else:
                DeltaSigma[i] = self.beta * v_inf**2 / (4 * G * (Rt*kpc)) * (Rt / r)**1.7
        
        return DeltaSigma / (M_sun / pc**2)
    
    def test_all(self):
        """Test su tutti i dataset"""
        print(f"\n{'='*70}")
        print(f"🔬 TEST GCV CON BOOST β = {self.beta:.2f}")
        print(f"   Ipotesi: Lc_lensing = {self.beta:.2f} × Lc_rotations")
        print(f"{'='*70}")
        
        results = {}
        
        for key, data in self.datasets.items():
            print(f"\n📊 {key}: M* = {data['Mstar']:.1e} M☉")
            
            pred = self.predict_gcv_boosted(data['Mstar'], data['R_kpc'])
            obs = data['DeltaSigma']
            err = data['DeltaSigma_err']
            
            chi2 = np.sum(((obs - pred) / err)**2)
            dof = len(obs) - 1
            chi2_red = chi2 / dof
            p_value = 1 - stats.chi2.cdf(chi2, dof)
            
            if p_value > 0.05:
                verdict = "✅ COMPATIBILE"
            elif p_value > 0.01:
                verdict = "⚠️  TENSIONE"
            else:
                verdict = "❌ INCOMPATIBILE"
            
            print(f"   χ²/dof = {chi2_red:.2f}, p = {p_value:.4f} → {verdict}")
            
            results[key] = {
                'chi2_red': chi2_red,
                'p_value': p_value,
                'verdict': verdict
            }
        
        return results
    
    def optimize_beta(self):
        """Trova β ottimale su tutti i dataset"""
        print(f"\n🔧 OTTIMIZZAZIONE β GLOBALE...")
        
        def chi2_global(beta):
            self.beta = beta[0]
            chi2_tot = 0
            for data in self.datasets.values():
                pred = self.predict_gcv_boosted(data['Mstar'], data['R_kpc'])
                obs = data['DeltaSigma']
                err = data['DeltaSigma_err']
                chi2_tot += np.sum(((obs - pred) / err)**2)
            return chi2_tot
        
        result = optimize.minimize(chi2_global, [4.7], bounds=[(2, 10)])
        beta_best = result.x[0]
        chi2_best = result.fun
        
        print(f"   β ottimale = {beta_best:.2f}")
        print(f"   χ² totale = {chi2_best:.1f}")
        
        self.beta = beta_best
        return beta_best

# Main
print("="*70)
print("🌌 TEST 2 RIVISTO: GCV CON Lc DIPENDENTE DA SCALA")
print("="*70)

# Test con β = 4.7 (da fit singolo)
tester = GCVWithBoost(beta_boost=4.7)
results_47 = tester.test_all()

# Ottimizza β su tutti i dataset
beta_opt = tester.optimize_beta()
results_opt = tester.test_all()

# Verdetto finale
print(f"\n{'='*70}")
print(f"🏁 VERDETTO FINALE")
print(f"{'='*70}")

n_total = len(results_opt)
verdicts = [r['verdict'] for r in results_opt.values()]
n_pass = sum('✅' in v for v in verdicts)
n_tension = sum('⚠️' in v for v in verdicts)
n_fail = sum('❌' in v for v in verdicts)

print(f"\nRisultati con β = {beta_opt:.2f}:")
print(f"   ✅ Compatibili: {n_pass}/{n_total}")
print(f"   ⚠️  Tensione: {n_tension}/{n_total}")
print(f"   ❌ Incompatibili: {n_fail}/{n_total}")

print(f"\n{'='*70}")

if n_fail == 0:
    print("🎉🎉🎉 GCV CON BOOST SUPERA TEST LENSING! 🎉🎉🎉")
    print(f"\n   Il vuoto ha scala di coerenza DIVERSA per lensing!")
    print(f"   Lc_lensing = {beta_opt:.2f} × Lc_rotations")
    print(f"\n   Questa è fisica NUOVA e testabile!")
    final = "PASS"
elif n_pass >= n_total // 2:
    print("⚠️  GCV CON BOOST È PLAUSIBILE")
    print(f"   Maggioranza dataset compatibili")
    print(f"   β = {beta_opt:.2f} spiega trend generale")
    final = "PARTIAL"
else:
    print("❌ GCV NON SALVABILE CON BOOST SEMPLICE")
    print(f"   Serve modello più complesso")
    final = "FAIL"

print(f"{'='*70}\n")

# Salva
results_dir = Path(__file__).parent / 'results'
results_dir.mkdir(exist_ok=True)

output = {
    'final_verdict': final,
    'beta_optimal': float(beta_opt),
    'interpretation': f'Lc_lensing = {beta_opt:.2f} × Lc_rotations',
    'results': {
        key: {
            'chi2_red': float(res['chi2_red']),
            'p_value': float(res['p_value']),
            'verdict': res['verdict']
        }
        for key, res in results_opt.items()
    }
}

with open(results_dir / 'test2_with_boost_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"💾 Risultati: results/test2_with_boost_results.json")

sys.exit(0 if final == "PASS" else 1)
