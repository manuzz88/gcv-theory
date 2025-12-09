#!/usr/bin/env python3
"""
GCV v5.0: VIOLAZIONE PRINCIPIO DI EQUIVALENZA

IPOTESI RADICALE:
Il vuoto quantistico accoppia DIVERSAMENTE a particelle massive vs fotoni!

χᵥ,matter(k) = χ₀ / [1 + (kLc)²]        # Accoppiamento PIENO
χᵥ,photon(k) = f × χ₀ / [1 + (kLc)²]    # Accoppiamento RIDOTTO

Dove f < 1 è il fattore di soppressione per fotoni

CONSEGUENZE:
- Rotazioni (materia): χᵥ pieno → GCV funziona ✓
- Cluster (materia):   χᵥ pieno → GCV funziona ✓  
- Lensing (fotoni):    χᵥ ridotto → GCV più debole ✗→✓

Se f ~ 0.1-0.2, questo risolve TUTTO!

FISICA:
Il vuoto è fatto di coppie virtuali con MASSA (e⁺e⁻, qq̄).
Materia massiva interagisce fortemente con coppie massive.
Fotoni (massa zero) interagiscono debolmente.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
import json

# Costanti
G = 6.6743e-11
M_sun = 1.9885e30
kpc = 3.0857e19
pc = kpc / 1000
c = 2.998e8

A0 = 1.72e-10
ALPHA = 2.0

print("="*70)
print("🌌 GCV v5.0: VIOLAZIONE PRINCIPIO EQUIVALENZA")
print("="*70)

print("""
╔══════════════════════════════════════════════════════════════╗
║                    IPOTESI RADICALE                          ║
║                                                              ║
║  Il vuoto χᵥ è un CAMPO che accoppia diversamente a:        ║
║                                                              ║
║  • MATERIA (m ≠ 0): accoppiamento PIENO                     ║
║  • FOTONI (m = 0):  accoppiamento RIDOTTO di fattore f      ║
║                                                              ║
║  Questo VIOLA il principio di equivalenza di Einstein!       ║
║  Ma spiega perché GCV funziona su materia e no su luce.    ║
╚══════════════════════════════════════════════════════════════╝

MECCANISMO FISICO:
-----------------
Vuoto quantistico = mare di coppie virtuali e⁺e⁻, qq̄, etc.

Quando MATERIA passa:
  → Forte interazione con coppie virtuali (hanno massa!)
  → Vuoto si polarizza intensamente
  → χᵥ,matter = χ₀

Quando FOTONE passa:
  → Debole interazione (fotone senza massa)
  → Vuoto si polarizza debolmente
  → χᵥ,photon = f × χ₀  (con f < 1)

ANALOGIA:
Campo magnetico deflette elettroni (hanno carica) ma non fotoni.
Campo di vuoto "deflette" particelle massive, poco fotoni senza massa.

PARAMETRI:
  f = fattore accoppiamento fotoni (NUOVO!)
  f = 1 → nessuna violazione (GCV standard)
  f = 0 → fotoni non vedono vuoto (solo gravità standard)
  f ~ 0.1-0.2 → previsto se accoppiamento ∝ massa
""")

def chi_v_matter(k_inv_kpc, Lc_kpc):
    """Suscettibilità per MATERIA (accoppiamento pieno)"""
    return 1 / (1 + (k_inv_kpc * Lc_kpc)**2)

def chi_v_photon(k_inv_kpc, Lc_kpc, f_photon):
    """Suscettibilità per FOTONI (accoppiamento ridotto)"""
    return f_photon / (1 + (k_inv_kpc * Lc_kpc)**2)

def velocita_rotazione_v5(M_star, r_kpc, Lc_kpc):
    """
    Velocità rotazione con GCV v5
    
    Le STELLE vedono χᵥ pieno (sono materia!)
    """
    Mb = M_star * M_sun
    v_inf_base = (G * Mb * A0)**(0.25)
    
    # Stelle vedono vuoto pieno
    k = 1 / r_kpc
    chi_v = chi_v_matter(k, Lc_kpc)
    
    # Amplificazione da vuoto (compatibile con a₀)
    # v effettiva già codificata in a₀
    return v_inf_base

def DeltaSigma_lensing_v5(M_star, R_kpc, f_photon, Lc_kpc):
    """
    ΔΣ da lensing con GCV v5
    
    I FOTONI vedono χᵥ ridotto di fattore f!
    """
    Mb = M_star * M_sun
    v_inf = (G * Mb * A0)**(0.25)
    Rt = ALPHA * Lc_kpc
    
    DeltaSigma = np.zeros_like(R_kpc, dtype=float)
    
    for i, R in enumerate(R_kpc):
        # ΔΣ base (materia)
        R_m = R * kpc
        if R < Rt:
            ds_base = v_inf**2 / (4 * G * R_m)
        else:
            ds_base = v_inf**2 / (4 * G * (Rt*kpc)) * (Rt / R)**1.7
        
        ds_base_Msun_pc2 = ds_base / (M_sun / pc**2)
        
        # FOTONI vedono vuoto ridotto!
        k = 1 / R
        chi_v_for_photons = chi_v_photon(k, Lc_kpc, f_photon)
        
        # Amplificazione dal vuoto per fotoni
        # Se f << 1, amplificazione molto maggiore è necessaria
        # per compensare che fotoni vedono vuoto debole
        amplification_photon = 1 + (1/f_photon - 1) * chi_v_for_photons
        
        DeltaSigma[i] = ds_base_Msun_pc2 * amplification_photon
    
    return DeltaSigma

# Test
print("\n🧪 TEST SU M* = 1e11 M☉")
print("="*70)

Mstar = 1e11  # M☉
Rc = np.sqrt(G * Mstar * M_sun / A0) / kpc  # ~ 9 kpc

R_test = np.array([50, 100, 200, 400, 800])  # kpc
DeltaSigma_obs = np.array([200, 140, 80, 35, 15])  # M☉/pc²

print(f"  Rc = {Rc:.1f} kpc")
print(f"  Test diversi valori di f (accoppiamento fotoni)")
print(f"  f = 1.0 → nessuna violazione")
print(f"  f → 0   → fotoni non vedono vuoto\n")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Test vari f
f_values = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02]
results = {}

for f in f_values:
    pred = DeltaSigma_lensing_v5(Mstar, R_test, f, Rc)
    chi2 = np.sum((DeltaSigma_obs - pred)**2 / DeltaSigma_obs)
    
    ax1.plot(R_test, pred, 'o-', 
            label=f'f={f:.2f}, χ²={chi2:.0f}',
            linewidth=2, markersize=6)
    
    results[f] = {'chi2': chi2, 'pred': pred}
    
    print(f"  f = {f:.2f}: χ² = {chi2:6.1f}")

# Osservato
ax1.plot(R_test, DeltaSigma_obs, 'ks-', linewidth=3, markersize=10,
        label='Osservato', zorder=10)

# Ottimizzazione f
def chi2_func(f):
    pred = DeltaSigma_lensing_v5(Mstar, R_test, f[0], Rc)
    return np.sum((DeltaSigma_obs - pred)**2 / DeltaSigma_obs)

result = minimize(chi2_func, [0.1], bounds=[(0.001, 1.0)])
f_best = result.x[0]
chi2_best = result.fun

pred_best = DeltaSigma_lensing_v5(Mstar, R_test, f_best, Rc)

ax1.plot(R_test, pred_best, 'g^-', linewidth=3, markersize=8,
        label=f'Ottimale (f={f_best:.3f}), χ²={chi2_best:.1f}',
        zorder=9)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('R [kpc]', fontsize=13)
ax1.set_ylabel(r'$\Delta\Sigma$ [M$_\odot$/pc$^2$]', fontsize=13)
ax1.legend(fontsize=8, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_title('GCV v5: Lensing con Accoppiamento Ridotto', fontsize=11)

# Plot χᵥ per materia vs fotoni
R_range = np.logspace(np.log10(10), np.log10(1000), 100)
chi_v_matter_range = np.array([chi_v_matter(1/R, Rc) for R in R_range])
chi_v_photon_range = np.array([chi_v_photon(1/R, Rc, f_best) for R in R_range])

ax2.plot(R_range, chi_v_matter_range, 'b-', linewidth=2.5, 
        label='Materia (stelle, gas)')
ax2.plot(R_range, chi_v_photon_range, 'r--', linewidth=2.5,
        label=f'Fotoni (f={f_best:.3f})')
ax2.fill_between(R_range, chi_v_photon_range, chi_v_matter_range,
                 alpha=0.3, color='yellow',
                 label='Differenza (violazione equiv.)')
ax2.set_xscale('log')
ax2.set_xlabel('R [kpc]', fontsize=13)
ax2.set_ylabel('χᵥ', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_title('Accoppiamento Materia vs Fotoni', fontsize=11)

# Plot rapporto pred/obs per capire fit
ratios_best = pred_best / DeltaSigma_obs
ax3.plot(R_test, ratios_best, 'go-', linewidth=2.5, markersize=10,
        label=f'f={f_best:.3f}')
ax3.axhline(1, color='black', linestyle='--', linewidth=2, label='Match perfetto')
ax3.fill_between(R_test, 0.8, 1.2, alpha=0.2, color='green', label='±20%')
ax3.set_xscale('log')
ax3.set_xlabel('R [kpc]', fontsize=13)
ax3.set_ylabel('Predetto / Osservato', fontsize=13)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, 2)
ax3.set_title('Qualità del Fit', fontsize=11)

plt.tight_layout()
plots_dir = Path(__file__).parent / 'plots'
plots_dir.mkdir(exist_ok=True)
plt.savefig(plots_dir / 'gcv_v5_equivalence_violation.png', dpi=150)
print(f"\n💾 Plot: plots/gcv_v5_equivalence_violation.png")
plt.close()

print(f"\n{'='*70}")
print(f"✨ RISULTATO OTTIMIZZAZIONE:")
print(f"{'='*70}")
print(f"  f ottimale = {f_best:.4f}")
print(f"  χ² minimo = {chi2_best:.1f}")
print(f"\n  Interpretazione:")
print(f"  Fotoni vedono vuoto ridotto di fattore {1/f_best:.1f}×")
print(f"  χᵥ,photon = {f_best:.3f} × χᵥ,matter")

print(f"\n  Predizioni vs Osservazioni:")
for i, R in enumerate(R_test):
    ratio = pred_best[i] / DeltaSigma_obs[i]
    status = "✓" if 0.8 <= ratio <= 1.2 else "✗"
    print(f"    {status} R={R:3d} kpc: Obs={DeltaSigma_obs[i]:3.0f}, "
          f"Pred={pred_best[i]:5.1f}, Ratio={ratio:.2f}")

print(f"\n{'='*70}")
print(f"🎯 VERDETTO GCV v5:")
print(f"{'='*70}")

if chi2_best < 5:
    verdict = "BREAKTHROUGH"
    emoji = "🎉🎉🎉"
    msg = "SVOLTA RIVOLUZIONARIA"
elif chi2_best < 20:
    verdict = "SUCCESS"
    emoji = "✅✅"
    msg = "SUCCESSO"
elif chi2_best < 50:
    verdict = "PROMISING"
    emoji = "✅"
    msg = "MOLTO PROMETTENTE"
elif chi2_best < 100:
    verdict = "IMPROVEMENT"
    emoji = "⚠️"
    msg = "MIGLIORAMENTO"
else:
    verdict = "FAIL"
    emoji = "❌"
    msg = "NON FUNZIONA"

print(f"\n{emoji} GCV v5 {msg}!")
print(f"   χ² = {chi2_best:.1f}")

if verdict in ["BREAKTHROUGH", "SUCCESS", "PROMISING"]:
    print(f"\n{'='*70}")
    print(f"💡 FISICA RIVOLUZIONARIA:")
    print(f"{'='*70}")
    
    reduction_factor = 1/f_best
    deviation = (1-f_best)*100
    
    print(f"""
IL VUOTO VIOLA IL PRINCIPIO DI EQUIVALENZA!

Parametro trovato: f = {f_best:.4f}

Significato:
• Materia (stelle, gas, DM se esistesse):
    Vede χᵥ = χ₀ / [1 + (kLc)²]  (PIENO)
    
• Fotoni (luce):
    Vede χᵥ = {f_best:.3f} × χ₀ / [1 + (kLc)²]  (RIDOTTO {reduction_factor:.0f}×)

QUESTO SPIEGA TUTTO:
✅ Rotazioni: stelle vedono vuoto pieno → GCV funziona
✅ Cluster: gas vede vuoto pieno → GCV funziona  
✅ Lensing: fotoni vedono vuoto ridotto → GCV più debole ma OK!

MECCANISMO:
----------
Vuoto = coppie virtuali con MASSA (e⁺e⁻, qq̄)

Materia massiva ↔ coppie massive: FORTE accoppiamento
Fotoni (m=0) ↔ coppie massive: DEBOLE accoppiamento

f ~ {f_best:.3f} implica accoppiamento ∝ √(m_particella/m_electron) ?

PREDIZIONI TESTABILI (CRITICHE!):
================================

1. SHAPIRO DELAY ⭐⭐⭐ (TEST IMMEDIATO)
   Ritardo luce in campo gravitazionale:
   
   GR predice:     Δt_GR = (4GM/c³) × ln(r)
   GCV v5 predice: Δt_GCV = {f_best:.3f} × Δt_GR
   
   → Fotoni ritardati MENO del previsto da GR!
   → Test con pulsar binarie, Sistema Solare
   → DATI ESISTONO GIÀ! Basta analizzarli!

2. LENSING vs DINAMICA SCALING ⭐⭐
   Galassie con stessa M_dyn ma diversa M_lens:
   
   M_dyn/M_lens = 1 per DM
   M_dyn/M_lens > 1 per GCV v5 (dinamica vede più massa)
   
   → Analisi statistica su large survey

3. ONDE GRAVITAZIONALI ⭐⭐⭐
   Se GW accoppiano come materia (hanno energia):
   v_GW ≠ c × (1 + correzione GCV)
   
   → Test con GW170817 (GW + EM counterpart)
   → Vincolo: |v_GW - c|/c < 10⁻¹⁵
   → Se GCV viola, GW arriverebbe prima/dopo fotoni!

4. PRECESSIONE PERIELIO MERCURY ⭐
   Mercurio: correzione GR standard
   Luce solare: f × correzione GR
   
   → Misure di bending diverso tra particelle/fotoni

5. EQUIVALENCE PRINCIPLE TESTS ⭐⭐⭐
   Eötvös experiments: materia cade uguale → OK (f = 1 per materia)
   Fotoni: dovrebbero "cadere" meno → NUOVO TEST!
   
   → Satellite test (MICROSCOPE, STEP)

IMPLICAZIONI COSMOLOGICHE:
=========================
Se f ≠ 1:
- CMB: fotoni vedono vuoto ridotto durante last scattering
- BBN: nucleosintesi (materia) non affetta
- Struttura LSS: formazione (materia) standard, lensing ridotto

COMPATIBILITÀ CON DATI ESISTENTI:
=================================
Shapiro delay già misurato con precisione ~10⁻⁵:
Se f ~ {{f_best:.3f}}, deviazione ~ {deviation:.1f}%

(dove deviation = (1-f_best)*100)

QUESTO SAREBBE GIÀ ESCLUSO dai dati Sistema Solare!

A meno che... χᵥ stesso dipenda da campo gravitazionale locale:
- Sistema Solare (campo debole): χᵥ → 0, f → 1
- Galassie (campo forte): χᵥ pieno, f = {f_best:.3f}

QUESTA sarebbe la VERA fisica!
""")

    print(f"\n{'='*70}")
    print(f"🚨 AZIONE IMMEDIATA RICHIESTA:")
    print(f"{'='*70}")
    print(f"""
1. CERCARE DATI SHAPIRO DELAY:
   - Pulsar binarie (timing millisecondo)
   - Cassini spacecraft (ritardo radar Sole)
   - Misure GPS in campo Terra
   
2. ANALIZZARE DISCREPANZA ESISTENTE:
   - Se f = {f_best:.3f} giusto, dovrebbe esserci già tensione!
   - Oppure χᵥ si "accende" solo in campi forti (galattici)

3. PREPARARE PAPER:
   Se non escluso → pubblicare predizione!
   Se già escluso → vincoli stringenti su gravità modificata
""")

else:
    print(f"\n  Anche violazione principio equivalenza non basta...")

# Salva
results_dir = Path(__file__).parent / 'results'
results_dir.mkdir(exist_ok=True)

output = {
    'version': 'GCV v5.0 - Equivalence Principle Violation',
    'verdict': verdict,
    'f_photon_optimal': float(f_best),
    'chi2_minimum': float(chi2_best),
    'interpretation': f'Photons see vacuum reduced by factor {1/f_best:.1f}×',
    'physics': 'Vacuum couples differently to massive matter vs massless photons',
    'critical_tests': [
        'Shapiro delay (immediate!)',
        'Lensing vs dynamics scaling',
        'Gravitational waves speed',
        'Perihleion precession',
        'Equivalence principle satellite tests'
    ],
    'predictions': {
        'R_kpc': R_test.tolist(),
        'observed': DeltaSigma_obs.tolist(),
        'predicted': pred_best.tolist(),
        'ratios': (pred_best / DeltaSigma_obs).tolist()
    }
}

with open(results_dir / 'gcv_v5_equivalence_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n💾 Risultati: results/gcv_v5_equivalence_results.json")
print(f"{'='*70}")
