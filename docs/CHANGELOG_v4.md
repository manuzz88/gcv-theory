# GCV v4.0 - Changelog

Data: 9 Dicembre 2025

## Novita Principali

### 1. Test Rigoroso su Dati Reali SDSS DR7

Testato GCV su dati REALI di weak lensing (non interpolati):
- 19 punti dati da Mandelbaum et al. 2006
- Matrice di covarianza completa
- Confronto equo con LCDM (NFW + baryons)

### 2. Derivazione Teorica di alpha_lens = 0.5

Scoperta fondamentale: l'esponente per il lensing DERIVA MATEMATICAMENTE dalla definizione di Delta Sigma:

- Sigma scala come (1 + chi_v)^1.0
- Delta Sigma scala come (1 + chi_v)^0.5

Questo NON e un parametro libero, ma una conseguenza della fisica!

### 3. GCV v2.3 - Modello Unificato

Nuovo modello che fitta simultaneamente rotation curves e lensing:

Parametri:
- A0 = 0.50
- beta = 0.63
- alpha_lens = 0.50 (derivato)

Risultati:
- Rotation curves: MAPE = 14.5% (eccellente, come MOND)
- Lensing: Delta AIC = +12 vs LCDM

### 4. Nuovi Script di Test

- 20_definitive_sdss_test.py: Test su dati reali
- 24_theoretical_derivation_alpha_lens.py: Derivazione di alpha
- 25_delta_sigma_analysis.py: Analisi Sigma vs Delta Sigma
- 29_theoretical_investigation.py: Investigazione teorica
- 31_gcv_v23_unified.py: Modello unificato

## Risultati Chiave

| Test | Risultato |
|------|-----------|
| Rotation curves (SPARC) | MAPE = 14.5% |
| Lensing (SDSS) | Delta AIC = +12 vs LCDM |
| Derivazione alpha | 0.5 (teorico) |

## Interpretazione

GCV v2.3 ha lo stesso problema di MOND:
- Funziona bene su scale galattiche
- Ha difficolta su scale piu grandi (lensing)

La differenza con MOND: GCV ha una derivazione teorica per alpha_lens.

## File Modificati

- gcv_gpu_tests/lensing/: 12 nuovi script
- gcv_gpu_tests/results/: Risultati JSON
- gcv_gpu_tests/plots/: Nuovi grafici

## Prossimi Passi

1. Derivare alpha_lens da principi primi (metrica)
2. Testare su altri dataset (DES, KiDS)
3. Esplorare dipendenza da ambiente
