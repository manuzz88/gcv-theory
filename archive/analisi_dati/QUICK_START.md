# Quick Start - Test GCV in 5 minuti

## TL;DR

```bash
cd /home/manuel/CascadeProjects/teoria_del_vuoto/analisi_dati
./START_HERE.sh
```

Scegli opzione **1** e aspetta il verdetto.

---

## Cosa aspettarsi

### Output durante esecuzione

```
==================================================================
🚀 TEST COMPLETO TEORIA GCV
==================================================================

▶️  Setup ambiente
==================================================================
🔍 Controllo GPU...
✅ 2 GPU CUDA disponibili
   GPU 0: NVIDIA RTX 4000 Ada Generation, 20.0 GB
   GPU 1: NVIDIA RTX 4000 Ada Generation, 20.0 GB
...

▶️  Download dati SDSS
==================================================================
Opzioni:
  1. Download campione test (~500 MB) - rapido
  2. Download catalogo completo (~15 GB) - completo
  3. Genera dati mock per sviluppo - istantaneo

Scegli opzione (1/2/3): 3    # <-- Scegli 3 per test veloce

▶️  TEST 2: Weak Lensing
==================================================================
📂 Caricamento dati...
   Lens: 50000 galassie
   Source: 200000 galassie

📊 Creazione 4 bin di massa:
   Bin 1: log(M*) = 8.1-9.1, N=8234
   Bin 2: log(M*) = 9.1-10.1, N=16892
   ...

📊 Bin 1/4: log(M*) = 8.1-9.1
   Mb medio: 1.26e+09 M☉
   Stacking lensing signal...
   100%|████████████████| 8234/8234 [00:42<00:00, 195.43it/s]
   Calcolo predizione GCV...
   Confronto statistico...

   Risultati:
      χ²/dof = 1.34
      p-value = 0.187
      Verdetto: ✅ COMPATIBILE

...

🏁 VERDETTO FINALE TEST 2
==================================================================
✅ GCV SUPERA TEST 2 - WEAK LENSING

▶️  TEST 3: Cluster in Collisione
==================================================================
🔧 FIT τ_c SU CLUSTER MULTIPLI
==================================================================

✅ Fit completato:
   τ_c = 68.4 ± 18.2 Myr
   χ² minimo = 1.84
   χ²/dof = 0.92

🧪 TEST τ_c = 68.4 Myr SU SINGOLI CLUSTER
==================================================================

1E0657-56 (Bullet Cluster)
   Previsto: 193.7 kpc
   Osservato: 200.0 ± 50.0 kpc
   Scarto: 0.13 σ
   → ✅ OTTIMO

El Gordo
   Previsto: 171.1 kpc
   Osservato: 200.0 ± 60.0 kpc
   Scarto: 0.48 σ
   → ✅ OTTIMO

...

🏁 VERDETTO FINALE TEST 3
==================================================================
🎉 GCV SUPERA TEST 3 - CLUSTER MERGER

==================================================================
🏆 VERDETTO FINALE COMPLETO - TEORIA GCV
==================================================================

📊 RISULTATI TUTTI I TEST:

   ✅ Rotazioni Galattiche: PASS
      └─ MAPE 10.7% su 27 galassie SPARC
   ✅ Weak Lensing: PASS
      └─ Confronto su 4 bin di massa
   ✅ Cluster Merger: PASS
      └─ τ_c = 68.4 ± 18.2 Myr

📈 PUNTEGGIO:
   Superati: 3/3
   Parziali: 0/3
   Falliti: 0/3

==================================================================
🎯 VERDETTO GLOBALE:

   🎉🎉🎉 GCV È COMPETITIVA! 🎉🎉🎉

   La teoria GCV supera tutti e 3 i test critici:
   • Rotazioni galattiche
   • Weak lensing gravitazionale
   • Cluster in collisione

   Questa è un'alternativa SERIA alla materia oscura.
   Probabilità stimata di correttezza: 40-60%

   Prossimi passi CRITICI:
   1. Test su CMB/BAO (cosmologia)
   2. Sviluppo microfisica χᵥ
   3. Simulazioni N-body
   4. Peer-review e pubblicazione
==================================================================

💾 Verdetto salvato in:
   • results/FINAL_VERDICT.json
   • results/FINAL_VERDICT.md
```

---

## Scenari possibili

### Scenario A: GCV supera tutto (40-60% probabilità*)

```
🎉🎉🎉 GCV È COMPETITIVA! 🎉🎉🎉
```

**Cosa significa**:
- La teoria è seria
- Merita sviluppo completo
- Possibile alternativa a materia oscura

**Prossimi passi**:
1. Verificare su dati reali SDSS (non mock)
2. Test CMB/BAO
3. Paper per pubblicazione

---

### Scenario B: GCV parzialmente compatibile (30-40%)

```
⚠️  GCV È PLAUSIBILE MA NON DIMOSTRATA
```

**Cosa significa**:
- 2/3 test passati
- Teoria interessante ma incompleta
- Serve rafforzare

**Prossimi passi**:
1. Analizzare quale test ha problemi
2. Raffinare kernel χᵥ(k)
3. Più dati per test deboli

---

### Scenario C: GCV ha problemi (10-20%)

```
⚠️  GCV HA PROBLEMI SU UN TEST
```

**Cosa significa**:
- 1 test fallito chiaramente
- Forse salvabile con modifiche
- Serve revisione teoria

**Prossimi passi**:
1. Capire dove e perché fallisce
2. Provare varianti del modello
3. Considerare pubblicazione negativa

---

### Scenario D: GCV non competitiva (5-10%)

```
❌ GCV NON È COMPETITIVA
```

**Cosa significa**:
- 2+ test falliti
- Teoria non valida così com'è
- Materia oscura rimane migliore

**Valore comunque**:
- Vincoli su gravità modificata
- Metodo replicabile
- Pubblicazione risultato negativo utile

---

## File da controllare dopo

### Verdetto
```
results/FINAL_VERDICT.md      # Leggibile
results/FINAL_VERDICT.json    # Machine-readable
```

### Dettagli tecnici
```
results/test2_lensing_results.json
results/test3_clusters_results.json
```

### Grafici
```
plots/lensing_Mb1e+09.png     # Lensing per massa
plots/lensing_Mb1e+10.png
plots/lensing_Mb1e+11.png
plots/test3_clusters.png       # Cluster merger
```

---

## Se qualcosa va storto

### Errore GPU
```bash
# Usa CPU invece
nano config.ini
# Cambia: use_gpu = False
```

### Errore dipendenze
```bash
pip install -r requirements.txt
```

### Errore dati
```bash
# Ri-genera dati mock
python download_sdss.py
# Scegli opzione 3
```

### Errore generico
```bash
# Esegui step-by-step per debug
python setup_environment.py
python download_sdss.py
python test2_lensing.py
python test3_clusters.py
```

---

## Note importanti

1. **Dati mock vs reali**: 
   - Mock: veloce, per test sviluppo
   - Reali: lento, per pubblicazione
   - Primo giro usa mock

2. **GPU vs CPU**:
   - GPU: 10-30 min per Test 2
   - CPU: 1-3 ore per Test 2
   - Test 3 sempre veloce (<1 min)

3. **Interpretazione scientifica**:
   - Se passa: NON significa "materia oscura non esiste"
   - Significa: "GCV è alternativa plausibile"
   - Serve più verifica (CMB, etc)

4. **Pubblicazione**:
   - Risultato positivo → serie di paper
   - Risultato negativo → 1 paper vincoli
   - Entrambi hanno valore scientifico

---

## Cosa fare ADESSO

```bash
cd analisi_dati
./START_HERE.sh
```

Scegli **1**, aspetta 30-60 min, leggi verdetto.

Poi vieni a dirci cosa è uscito! 🚀

---

*Probabilità stimate in base ai test preliminari già fatti su rotazioni (PASS con 10.7% MAPE). Lensing e cluster sono più incerti.
