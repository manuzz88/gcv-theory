#!/usr/bin/env python3
"""
TEST 3: CLUSTER MERGER (Bullet Cluster e simili)

Analisi:
1. Carica dati pubblicati su velocità e offset
2. Calcola lag previsto da GCV con τc
3. Fit τc su multipli sistemi
4. Verdetto su compatibilità
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
import json

# Costanti
MYR_TO_SEC = 3.1557e13  # secondi in 1 Myr
KPC_TO_M = 3.0857e19  # metri in 1 kpc

class ClusterMergerAnalysis:
    """Analisi cluster merger per test GCV"""
    
    def __init__(self):
        """Inizializza con dati pubblicati"""
        
        # Database merger osservati
        # Dati da letteratura: Clowe+2006, Menanteau+2012, etc.
        
        self.clusters = {
            'Bullet': {
                'name': '1E0657-56 (Bullet Cluster)',
                'vrel_kms': 4500,  # km/s velocità impatto stimata
                'vrel_err': 700,
                'offset_kpc': 200,  # kpc separazione massa-gas osservata
                'offset_err': 50,
                'redshift': 0.296,
                'reference': 'Clowe et al. 2006',
                'notes': 'Caso iconico, post-merger ~150 Myr fa'
            },
            'El Gordo': {
                'name': 'ACT-CL J0102-4915 (El Gordo)',
                'vrel_kms': 2500,
                'vrel_err': 400,
                'offset_kpc': 200,
                'offset_err': 60,
                'redshift': 0.870,
                'reference': 'Menanteau et al. 2012',
                'notes': 'Merger massiccio a z alto'
            },
            'MACS_J0025': {
                'name': 'MACS J0025.4-1222',
                'vrel_kms': 3000,
                'vrel_err': 500,
                'offset_kpc': 150,
                'offset_err': 40,
                'redshift': 0.586,
                'reference': 'Bradač et al. 2008',
                'notes': 'Baby Bullet, merger recente'
            }
        }
        
        print("="*60)
        print("🌌 CLUSTER MERGER DATABASE")
        print("="*60)
        for key, cluster in self.clusters.items():
            print(f"\n{cluster['name']}")
            print(f"  v_rel: {cluster['vrel_kms']} ± {cluster['vrel_err']} km/s")
            print(f"  Offset: {cluster['offset_kpc']} ± {cluster['offset_err']} kpc")
            print(f"  Ref: {cluster['reference']}")
    
    def predict_offset(self, vrel_kms, tau_c_Myr):
        """
        Predizione GCV dell'offset massa-gas
        
        Δx = v_rel × τ_c
        
        Parameters
        ----------
        vrel_kms : float
            Velocità relativa [km/s]
        tau_c_Myr : float
            Tempo di risposta del vuoto [Myr]
        
        Returns
        -------
        offset_kpc : float
            Offset previsto [kpc]
        """
        # Conversioni
        vrel_ms = vrel_kms * 1000  # m/s
        tau_c_s = tau_c_Myr * MYR_TO_SEC  # s
        
        # Offset
        offset_m = vrel_ms * tau_c_s
        offset_kpc = offset_m / KPC_TO_M
        
        return offset_kpc
    
    def chi2_single_cluster(self, tau_c_Myr, cluster_data):
        """
        Chi-quadro per singolo cluster
        
        Parameters
        ----------
        tau_c_Myr : float
            Tempo risposta GCV [Myr]
        cluster_data : dict
            Dati cluster
        
        Returns
        -------
        chi2 : float
            Contributo al chi-quadro
        """
        # Predizione
        offset_pred = self.predict_offset(
            cluster_data['vrel_kms'],
            tau_c_Myr
        )
        
        # Osservato
        offset_obs = cluster_data['offset_kpc']
        offset_err = cluster_data['offset_err']
        
        # Chi2
        chi2 = ((offset_pred - offset_obs) / offset_err)**2
        
        return chi2
    
    def chi2_all_clusters(self, tau_c_Myr):
        """
        Chi-quadro totale su tutti i cluster
        
        Returns
        -------
        chi2_tot : float
            Chi-quadro totale
        """
        chi2_tot = 0
        
        for cluster_data in self.clusters.values():
            chi2_tot += self.chi2_single_cluster(tau_c_Myr, cluster_data)
        
        return chi2_tot
    
    def fit_tau_c(self):
        """
        Fit ottimale di τ_c su tutti i cluster
        
        Returns
        -------
        tau_c_best : float
            Valore ottimale τ_c [Myr]
        tau_c_err : float
            Errore su τ_c [Myr]
        """
        print("\n" + "="*60)
        print("🔧 FIT τ_c SU CLUSTER MULTIPLI")
        print("="*60)
        
        # Minimizzazione chi2
        result = minimize(
            self.chi2_all_clusters,
            x0=[75.0],  # Guess iniziale τ_c = 75 Myr
            bounds=[(10, 200)],  # Range plausibile
            method='L-BFGS-B'
        )
        
        tau_c_best = result.x[0]
        chi2_min = result.fun
        
        # Errore da curvatura (approssimato)
        # Δχ² = 1 per 1σ
        tau_test = np.linspace(tau_c_best - 30, tau_c_best + 30, 100)
        chi2_curve = [self.chi2_all_clusters(t) for t in tau_test]
        
        # Trova τ dove χ² = χ²_min + 1
        idx_1sigma = np.where(np.array(chi2_curve) <= chi2_min + 1)[0]
        if len(idx_1sigma) > 1:
            tau_c_err = (tau_test[idx_1sigma[-1]] - tau_test[idx_1sigma[0]]) / 2
        else:
            tau_c_err = 20.0  # Fallback
        
        print(f"\n✅ Fit completato:")
        print(f"   τ_c = {tau_c_best:.1f} ± {tau_c_err:.1f} Myr")
        print(f"   χ² minimo = {chi2_min:.2f}")
        
        # dof = n_clusters - 1 parametro
        dof = len(self.clusters) - 1
        chi2_red = chi2_min / dof
        
        print(f"   χ²/dof = {chi2_red:.2f}")
        
        return tau_c_best, tau_c_err, chi2_red
    
    def test_individual_clusters(self, tau_c_Myr):
        """
        Test τ_c su ogni cluster individualmente
        
        Parameters
        ----------
        tau_c_Myr : float
            Valore τ_c da testare
        
        Returns
        -------
        results : dict
            Risultati per cluster
        """
        print(f"\n{'='*60}")
        print(f"🧪 TEST τ_c = {tau_c_Myr:.1f} Myr SU SINGOLI CLUSTER")
        print(f"{'='*60}")
        
        results = {}
        
        for key, cluster in self.clusters.items():
            print(f"\n{cluster['name']}")
            
            # Predizione
            offset_pred = self.predict_offset(
                cluster['vrel_kms'],
                tau_c_Myr
            )
            
            # Osservato
            offset_obs = cluster['offset_kpc']
            offset_err = cluster['offset_err']
            
            # Scarto in sigma
            sigma = abs(offset_pred - offset_obs) / offset_err
            
            # Verdetto
            if sigma < 1:
                verdict = "✅ OTTIMO"
            elif sigma < 2:
                verdict = "✅ BUONO"
            elif sigma < 3:
                verdict = "⚠️  ACCETTABILE"
            else:
                verdict = "❌ INCOMPATIBILE"
            
            print(f"   Previsto: {offset_pred:.1f} kpc")
            print(f"   Osservato: {offset_obs:.1f} ± {offset_err:.1f} kpc")
            print(f"   Scarto: {sigma:.2f} σ")
            print(f"   → {verdict}")
            
            results[key] = {
                'offset_pred': offset_pred,
                'offset_obs': offset_obs,
                'offset_err': offset_err,
                'sigma': sigma,
                'verdict': verdict
            }
        
        return results
    
    def plot_results(self, tau_c_best, tau_c_err, results, save=True):
        """
        Plot risultati: offset predetto vs osservato
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Offset predetto vs osservato
        cluster_names = [c['name'].split('(')[0].strip() 
                        for c in self.clusters.values()]
        
        offset_obs = [r['offset_obs'] for r in results.values()]
        offset_err = [r['offset_err'] for r in results.values()]
        offset_pred = [r['offset_pred'] for r in results.values()]
        
        x_pos = np.arange(len(cluster_names))
        
        ax1.errorbar(x_pos, offset_obs, yerr=offset_err,
                    fmt='o', label='Osservato', markersize=8,
                    color='black', capsize=5)
        ax1.plot(x_pos, offset_pred, 's', label='GCV', 
                markersize=10, color='red')
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(cluster_names, rotation=15, ha='right')
        ax1.set_ylabel('Offset massa-gas [kpc]', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'GCV: τ_c = {tau_c_best:.1f} ± {tau_c_err:.1f} Myr',
                     fontsize=12)
        
        # Plot 2: Scatter predetto vs osservato
        ax2.errorbar(offset_obs, offset_pred, 
                    xerr=offset_err, fmt='o',
                    markersize=8, color='blue', capsize=5)
        
        # Linea 1:1
        lim_min = min(min(offset_obs), min(offset_pred)) * 0.8
        lim_max = max(max(offset_obs), max(offset_pred)) * 1.2
        ax2.plot([lim_min, lim_max], [lim_min, lim_max],
                'k--', alpha=0.5, label='1:1')
        
        # Bande ±1σ, ±2σ
        offset_range = np.linspace(lim_min, lim_max, 100)
        err_mean = np.mean(offset_err)
        ax2.fill_between(offset_range,
                         offset_range - err_mean,
                         offset_range + err_mean,
                         alpha=0.2, color='green', label='±1σ')
        ax2.fill_between(offset_range,
                         offset_range - 2*err_mean,
                         offset_range + 2*err_mean,
                         alpha=0.1, color='yellow', label='±2σ')
        
        ax2.set_xlabel('Offset osservato [kpc]', fontsize=12)
        ax2.set_ylabel('Offset predetto GCV [kpc]', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        ax2.set_xlim(lim_min, lim_max)
        ax2.set_ylim(lim_min, lim_max)
        
        plt.tight_layout()
        
        if save:
            plots_dir = Path(__file__).parent / 'plots'
            plots_dir.mkdir(exist_ok=True)
            plt.savefig(plots_dir / 'test3_clusters.png',
                       dpi=150, bbox_inches='tight')
            print(f"\n💾 Plot salvato: plots/test3_clusters.png")
        
        plt.close()
    
    def final_verdict(self, results, chi2_red):
        """
        Verdetto finale Test 3
        """
        print("\n" + "="*60)
        print("🏁 VERDETTO FINALE TEST 3")
        print("="*60)
        
        n_clusters = len(results)
        verdicts = [r['verdict'] for r in results.values()]
        
        n_good = sum('✅' in v for v in verdicts)
        n_acceptable = sum('⚠️' in v for v in verdicts)
        n_bad = sum('❌' in v for v in verdicts)
        
        print(f"\nRisultati su {n_clusters} cluster:")
        print(f"   ✅ Buoni: {n_good}/{n_clusters}")
        print(f"   ⚠️  Accettabili: {n_acceptable}/{n_clusters}")
        print(f"   ❌ Incompatibili: {n_bad}/{n_clusters}")
        print(f"\n   χ²/dof = {chi2_red:.2f}")
        
        print(f"\n{'='*60}")
        
        if n_bad == 0 and chi2_red < 2:
            print("🎉 GCV SUPERA TEST 3 - CLUSTER MERGER")
            print(f"   Un unico τ_c spiega tutti i merger osservati")
            final = "PASS"
        elif n_bad <= 1 and chi2_red < 3:
            print("⚠️  GCV PARZIALMENTE COMPATIBILE")
            print("   Alcuni scarti ma globalmente plausibile")
            final = "PARTIAL"
        else:
            print("❌ GCV FALLISCE TEST 3 - CLUSTER MERGER")
            print("   Non riesce a spiegare offset con τ_c unico")
            final = "FAIL"
        
        print(f"{'='*60}\n")
        
        return final

def main():
    """Analisi completa Test 3"""
    
    print("\n" + "="*60)
    print("🚀 TEST 3: CLUSTER IN COLLISIONE")
    print("="*60)
    
    # Inizializza
    analyzer = ClusterMergerAnalysis()
    
    # Fit τ_c ottimale
    tau_c_best, tau_c_err, chi2_red = analyzer.fit_tau_c()
    
    # Test su singoli cluster
    results = analyzer.test_individual_clusters(tau_c_best)
    
    # Plot
    analyzer.plot_results(tau_c_best, tau_c_err, results)
    
    # Verdetto
    final_verdict = analyzer.final_verdict(results, chi2_red)
    
    # Salva risultati
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    output = {
        'final_verdict': final_verdict,
        'tau_c_best_Myr': tau_c_best,
        'tau_c_err_Myr': tau_c_err,
        'chi2_reduced': chi2_red,
        'individual_results': {
            key: {
                'offset_pred_kpc': r['offset_pred'],
                'offset_obs_kpc': r['offset_obs'],
                'sigma_discrepancy': r['sigma'],
                'verdict': r['verdict']
            }
            for key, r in results.items()
        }
    }
    
    with open(results_dir / 'test3_clusters_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 Risultati salvati in: results/test3_clusters_results.json")
    
    return 0 if final_verdict == "PASS" else 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
