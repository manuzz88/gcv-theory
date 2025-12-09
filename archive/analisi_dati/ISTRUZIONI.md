# Istruzioni: Esecuzione Test Definitivi GCV

## Obiettivo

Eseguire Test 2 (Lensing) e Test 3 (Cluster) per determinare se la **GCV è competitiva** con la materia oscura.

---

## Setup Rapido

### 1. Installa dipendenze

```bash
cd /home/manuel/CascadeProjects/teoria_del_vuoto/analisi_dati

# Crea ambiente virtuale (raccomandato)
python3 -m venv venv
source venv/bin/activate

# Installa pacchetti
pip install -r requirements.txt
```

**Nota GPU**: Se hai CUDA 12.x installato, cupy si installerà automaticamente. Altrimenti usa CPU (più lento ma funziona).

---

## Opzione A: Esecuzione Completa Automatica (RACCOMANDATO)

### Un solo comando per tutto

```bash
python run_all_tests.py
```

Questo script:
1. ✅ Controlla GPU e storage
2. 📥 Scarica/genera dati
3. 🧪 Esegue Test 2 (Lensing)
4. 🧪 Esegue Test 3 (Cluster)
5. 📊 Da verdetto finale

**Tempo**: 30-60 minuti

**Output**:
- `results/FINAL_VERDICT.json` - Verdetto tecnico
- `results/FINAL_VERDICT.md` - Verdetto leggibile
- `plots/*.png` - Grafici risultati

---

## Opzione B: Esecuzione Step-by-Step (per debug)

### Step 1: Setup

```bash
python setup_environment.py
```

Controlla:
- ✅ GPU disponibili
- 💾 Storage sufficiente
- 📦 Dipendenze installate

### Step 2: Download dati

```bash
python download_sdss.py
```

**Opzioni**:
1. **Campione test** (~500 MB, veloce) - **RACCOMANDATO per primo test**
2. Catalogo completo (~15 GB, lento)
3. Dati mock (istantaneo, per sviluppo)

Per primo giro scegli **Opzione 1** o **3**.

### Step 3: Test Lensing

```bash
python test2_lensing.py
```

Calcola ΔΣ(R) su 4 bin di massa e confronta con GCV.

**Output**:
- `results/test2_lensing_results.json`
- `plots/lensing_Mb*.png` (un plot per bin)

**Tempo**: 10-30 minuti (dipende da GPU)

### Step 4: Test Cluster

```bash
python test3_clusters.py
```

Fit τ_c su Bullet, El Gordo, MACS J0025.

**Output**:
- `results/test3_clusters_results.json`
- `plots/test3_clusters.png`

**Tempo**: < 1 minuto (calcoli analitici)

### Step 5: Verdetto finale

```bash
python run_all_tests.py
```

(Anche se hai già eseguito i test, questo leggerà i risultati e darà verdetto)

---

## Interpretazione Risultati

### Verdetti possibili

#### 🎉 FULLY_COMPETITIVE
- GCV supera tutti e 3 i test
- **Teoria seria** alternativa a DM
- Prossimi passi: CMB, pubblicazione

#### ⚠️ PLAUSIBLE
- GCV passa 2/3 test
- Teoria interessante ma da rafforzare
- Servono più dati

#### ⚠️ NEEDS_REVISION
- GCV passa 1-2/3 test
- Problemi ma forse risolvibili
- Serve modificare kernel/parametri

#### ❌ NOT_COMPETITIVE
- GCV fallisce 2+/3 test
- Teoria non valida nella forma attuale
- Materia oscura rimane migliore

---

## File Output Chiave

### Risultati

```
results/
├── FINAL_VERDICT.json         # Verdetto finale (machine-readable)
├── FINAL_VERDICT.md            # Verdetto finale (human-readable)
├── test2_lensing_results.json  # Dettagli Test 2
└── test3_clusters_results.json # Dettagli Test 3
```

### Plot

```
plots/
├── lensing_Mb1e+08.png         # Lensing bin 1
├── lensing_Mb1e+09.png         # Lensing bin 2
├── lensing_Mb1e+10.png         # Lensing bin 3
├── lensing_Mb1e+11.png         # Lensing bin 4
└── test3_clusters.png          # Cluster merger
```

### Dati

```
data/
├── sdss/
│   ├── mock_lens_catalog.fits    # Catalogo lens (mock o reale)
│   └── mock_source_catalog.fits  # Catalogo source (mock o reale)
└── clusters/
    └── (vuoto, dati da letteratura hardcoded)
```

---

## Configurazione Avanzata

Modifica `config.ini` per cambiare parametri:

```ini
[ANALYSIS]
# Bin di massa
mass_bins = 8, 9, 10, 11, 12  # log10(M*/Msun)

# Range radiale
radii_min_kpc = 30
radii_max_kpc = 1000
n_radii = 20

# Parametri GCV
a0 = 1.72e-10  # m/s²
alpha = 2.0    # Rt = alpha * Rc

[COMPUTING]
use_gpu = True
n_workers = 8
batch_size = 10000
```

---

## Troubleshooting

### GPU non rilevata

```
⚠️  GPU non disponibile: No module named 'cupy'
```

**Soluzione**: Installa cupy o continua con CPU:
```bash
pip install cupy-cuda12x  # Per CUDA 12
# oppure
# Modifica config.ini: use_gpu = False
```

### Storage insufficiente

```
⚠️  Meno di 50 GB liberi
```

**Soluzione**: Usa dati mock (Opzione 3 in download):
```bash
python download_sdss.py
# Scegli: 3
```

### File dati non trovati

```
❌ File dati non trovati!
```

**Soluzione**: Esegui prima download:
```bash
python download_sdss.py
```

### Errore import astropy

```
ModuleNotFoundError: No module named 'astropy'
```

**Soluzione**:
```bash
pip install -r requirements.txt
```

---

## FAQ

### Q: Quanto tempo serve?

**A**: 
- Setup: 10 min
- Download mock: 1 min
- Test 2: 10-30 min (GPU) / 1-2 ore (CPU)
- Test 3: < 1 min
- **Totale**: 30-60 min con GPU, 2-3 ore con CPU

### Q: Posso usare solo CPU?

**A**: Sì, imposta `use_gpu = False` in `config.ini`. Più lento ma funziona.

### Q: I dati mock sono validi?

**A**: Per **test di sviluppo** sì. Per **risultati pubblicabili** servono dati reali SDSS (Opzione 2 in download).

### Q: Cosa faccio se GCV passa tutti i test?

**A**:
1. Verifica su dati reali (non mock)
2. Estendi a più galassie/cluster
3. Test CMB/BAO (Fase 2)
4. Scrivi paper per peer-review

### Q: Cosa faccio se GCV fallisce?

**A**:
1. Analizza quale test fallisce
2. Verifica se è problema di dati o teoria
3. Considera modifiche a kernel χᵥ(k)
4. Se non salvabile: pubblica risultato negativo (comunque utile!)

---

## Contatti e Supporto

**Bug o problemi**: Apri issue nel progetto

**Collaborazioni**: Se risultati promettenti, considera collaborazione con esperti lensing/cluster

**Pubblicazione**: Se GCV passa test, prepara paper per ApJ/MNRAS

---

## Checklist Pre-Esecuzione

Prima di eseguire `run_all_tests.py`, verifica:

- [ ] Python 3.8+ installato
- [ ] GPU CUDA funzionante (opzionale ma raccomandato)
- [ ] Almeno 5 GB storage libero (mock) o 50 GB (dati reali)
- [ ] Dipendenze installate (`pip install -r requirements.txt`)
- [ ] Directory `analisi_dati` è quella corrente

Se tutto ok:

```bash
python run_all_tests.py
```

E aspetta il verdetto! 🚀

---

## Nota Finale

Questo è un test **scientifico rigoroso**. Il verdetto che otterrai sarà basato su:
- Confronto quantitativo con dati reali (o mock realistici)
- Statistiche chi-quadro e p-value
- Criteri falsificabili

**Non è garantito che GCV passi**. La scienza richiede onestà: se fallisce, pubblichiamo comunque il risultato negativo.

Buona fortuna! 🎲🔬
