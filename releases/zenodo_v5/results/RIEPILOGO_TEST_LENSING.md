# Riepilogo Test Lensing GCV

Data: 9 Dicembre 2025

## Obiettivo

Testare GCV su dati REALI di weak lensing (SDSS DR7) con confronto equo vs LCDM.

## Test Eseguiti

### Test 1: GCV con parametri da rotation curves
- Parametri: a0=1.8e-10, A0=1.16, gamma=0.06, beta=0.90
- Risultato: Chi2/dof = 3.6, Delta AIC = +35
- Verdetto: LCDM FAVORITA

### Test 2: GCV con parametri ottimizzati per lensing
- Ottimizzazione globale (Differential Evolution)
- Risultato: Chi2/dof = 1.04 (ottimizzazione), ma problemi di generalizzazione
- Verdetto: LCDM ancora favorita

### Test 3: Derivazione rigorosa (Poisson modificata)
- Derivazione completa: Poisson -> Phi -> Sigma -> Delta Sigma
- Risultato: Chi2/dof > 100
- Problema: La formula di chi_v non produce il segnale corretto
- Verdetto: LCDM FORTEMENTE FAVORITA

### Test 4: GCV ESTESO (unificazione lensing + rotation curves)
- Nuova formula: chi_v_lens = A_lens * chi_v_dyn^alpha_lens * (1 + R/R_lens)^delta
- Parametri ottimali:
  - alpha_lens = 0.42 (lensing vede sqrt di chi_v)
  - R_lens = 1000 kpc
  - delta = -0.59
  - A_lens = 0.78
- Risultato: Chi2/dof = 1.48, Delta AIC = 0.2
- Verdetto: GCV e LCDM EQUIVALENTI

## Risultato Principale

GCV puo' fittare il lensing SE si introduce una relazione tra chi_v dinamico e chi_v lensing:

    chi_v_lens ~ chi_v_dyn^0.42

Interpretazione fisica: il lensing e' sensibile a sqrt(chi_v) perche':
- Dinamica: v^2 ~ M_eff = M_b * (1 + chi_v)
- Lensing: Delta Sigma ~ sqrt(M_eff) per effetti di proiezione

## Implicazioni per la Teoria

1. GCV v2.1 con parametri da rotation curves NON fitta automaticamente il lensing
2. Serve un'estensione (4 parametri aggiuntivi) per unificare i due regimi
3. L'estensione ha una possibile interpretazione fisica (proiezione)
4. Con l'estensione, GCV e' EQUIVALENTE a LCDM

## Prossimi Passi

1. Derivare alpha_lens ~ 0.5 da principi primi (proiezione rigorosa)
2. Testare su altri dataset (DES, KiDS, HSC)
3. Verificare consistenza con cluster mergers
4. Ridurre il numero di parametri se possibile

## Conclusione

GCV rimane una teoria competitiva, ma richiede un'estensione per il lensing.
La buona notizia: l'estensione ha una possibile interpretazione fisica.
La sfida: derivare l'estensione da principi primi.

## File Generati

- 20_definitive_sdss_test.py: Test con parametri fissi
- 21_mcmc_lensing_gpu.py: Ottimizzazione parametri
- 22_rigorous_lensing_derivation.py: Derivazione rigorosa
- 23_gcv_lensing_extension.py: Modello esteso unificato
- plots/gcv_extended_unified.png: Plot risultati
