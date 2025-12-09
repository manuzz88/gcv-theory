# Indice Completo - Teoria GCV

## Navigazione rapida

- [📘 README principale](README.md) - Panoramica del progetto
- [📖 Glossario](GLOSSARIO.md) - Termini e definizioni
- [🎯 Sintesi finale](05_risultati/sintesi_finale.md) - Verdetto su competitività GCV

---

## 01. Concetti Base

Fondamenti cosmologici per capire il contesto della GCV.

### [Vuoto Cosmico](01_concetti_base/vuoto_cosmico.md)
- Definizione e caratteristiche
- Cosa contiene
- Ruolo per la GCV

### [Fluttuazioni Quantistiche](01_concetti_base/fluttuazioni_quantistiche.md)
- Principio di Heisenberg
- Effetto Casimir
- Connessione con energia del vuoto

### [Materia Oscura](01_concetti_base/materia_oscura.md)
- Problema osservativo
- Evidenze (rotazioni, lensing, cluster)
- Pro e contro del modello standard

### [Problema Rotazione Galattica](01_concetti_base/problema_rotazione_galattica.md)
- Sistema Solare vs Galassie
- Perché le stelle dovrebbero scappare
- Curve di rotazione piatte

---

## 02. Sviluppo Ipotesi

Come siamo arrivati alla formulazione della GCV.

### [Ipotesi Iniziali](02_sviluppo_ipotesi/ipotesi_iniziali.md)
5 idee alternative esplorate:
1. Fluido gravitazionale cosmico
2. Gravità modificata a grandi distanze
3. Universo ologramma gravitazionale
4. Gravità cooperativa delle stelle
5. Galassia come bolla in tensione

Perché nessuna era sufficiente.

### [Verso la GCV](02_sviluppo_ipotesi/verso_la_gcv.md)
- L'intuizione chiave
- Lo spazio non è neutro
- Comportamento scala-dipendente
- Differenza da MOND e materia oscura

---

## 03. Modello Matematico

Formulazione tecnica completa della teoria.

### [Formulazione GCV](03_modello_matematico/formulazione_gcv.md)

**Equazioni fondamentali**:
- Equazione modificata di Poisson
- Suscettibilità χᵥ(k) scala-dipendente
- Relazione Tully-Fisher: v∞⁴ = G·Mb·a₀
- Profilo con transizione (r⁻² → r⁻³)

**Parametri**:
- a₀ = 1.72 × 10⁻¹⁰ m/s²
- α ≈ 2 (transizione)
- τc ≈ 50-100 Myr (cluster)

**Predizioni**:
- Curve piatte senza DM
- Lensing ∝ √Mb
- Offset cluster con τc

---

## 04. Test Osservativi

Confronto quantitativo con dati reali.

### [Test 1: Rotazioni Galattiche](04_test_osservativi/test1_rotazioni_galattiche.md)

**Dataset**: 27 galassie SPARC

**Risultato**: ✅ SUPERATO
- MAPE: 10.7%
- Un parametro globale
- Match quasi perfetto per NGC 3198, NGC 5907

**Confronto**:
- GCV vs ΛCDM: pari per semplicità
- Predittivo: dato Mb → calcolo v∞

### [Test 2: Lensing Gravitazionale](04_test_osservativi/test2_lensing_gravitazionale.md)

**Dataset**: Stack SDSS (da confrontare)

**Risultato**: ⚠️ DA VERIFICARE
- SIS puro rigettato dai dati
- Versione con transizione compatibile in principio
- Serve confronto numerico con stack

**Predizione**:
- Forma: pendenza -1.0 → -1.7
- Ampiezza: ΔΣ ∝ √Mb

### [Test 3: Cluster Merger](04_test_osservativi/test3_cluster_merger.md)

**Dataset**: Bullet, El Gordo

**Risultato**: ⚠️ POSSIBILE
- Con τc ~ 50-100 Myr offset compatibili
- Serve verifica su 2-3 sistemi indipendenti
- Meccanismo: ritardo risposta vuoto

**Confronto**:
- DM collisionless: naturale
- GCV con τc: plausibile ma da dimostrare

---

## 05. Risultati

Sintesi, dati numerici e prossimi passi.

### [Sintesi Finale](05_risultati/sintesi_finale.md)

**Verdetto complessivo**: PLAUSIBILE ma NON ANCORA DIMOSTRATA

| Test | Stato | Competitività |
|------|-------|---------------|
| Rotazioni | ✅ Superato | ⭐⭐⭐⭐⭐ |
| Lensing | ⚠️ Da verificare | ⭐⭐⭐ |
| Cluster | ⚠️ Possibile | ⭐⭐⭐ |

**Stima probabilistica**:
- Supera tutti i test: 15-25%
- Supera rotazioni, fallisce altri: 40-50%
- Competitiva ma non migliore: 20-30%
- Rigettata: 10-15%

### [Dati Numerici](05_risultati/dati_numerici.md)

**Parametri ottimizzati** con valori precisi

**Tabelle complete**:
- 27 galassie SPARC (Vf osservato vs predetto)
- Scale caratteristiche (Rc, Rt per varie masse)
- Predizioni lensing (ΔΣ per 4 bin di massa)
- Lag cluster (Δx per varie vrel e τc)

**Codice Python** per calcoli base

### [Prossimi Passi](05_risultati/prossimi_passi.md)

**Roadmap completa** in 6 fasi:

**Fase 1** (3-6 mesi): Consolidamento test esistenti
- Estensione SPARC a 100+ galassie
- Stack lensing SDSS
- Validazione τc su 3 cluster

**Fase 2** (6-12 mesi): Test cosmologici
- Crescita strutture
- CMB/BAO

**Fase 3** (parallelo): Sviluppo teorico
- Microfisica di χᵥ
- Dinamica di τc

**Fase 4** (12-18 mesi): Predizioni uniche
- Fenomeni distinguibili da DM
- Simulazioni N-body

**Fase 5** (18-24 mesi): Pubblicazione
- Serie di 5 paper
- Conferenze
- Divulgazione

**Fase 6**: Piano di contingenza se fallisce

---

## Come usare questa documentazione

### Per lettura sequenziale

Ordine consigliato per comprendere tutto il progetto:

1. [README.md](README.md) - Panoramica
2. [Problema Rotazione Galattica](01_concetti_base/problema_rotazione_galattica.md) - Il problema
3. [Materia Oscura](01_concetti_base/materia_oscura.md) - Soluzione standard
4. [Verso la GCV](02_sviluppo_ipotesi/verso_la_gcv.md) - Nostra idea
5. [Formulazione GCV](03_modello_matematico/formulazione_gcv.md) - La teoria
6. [Test 1 Rotazioni](04_test_osservativi/test1_rotazioni_galattiche.md) - Primo test
7. [Sintesi Finale](05_risultati/sintesi_finale.md) - Verdetto

### Per accesso rapido

**Solo risultati numerici**:
→ [Dati Numerici](05_risultati/dati_numerici.md)

**Solo verdetto finale**:
→ [Sintesi Finale](05_risultati/sintesi_finale.md)

**Solo equazioni**:
→ [Formulazione GCV](03_modello_matematico/formulazione_gcv.md)

**Cosa fare dopo**:
→ [Prossimi Passi](05_risultati/prossimi_passi.md)

### Per approfondimenti

**Termini non chiari**:
→ [GLOSSARIO.md](GLOSSARIO.md)

**Contesto cosmologico**:
→ Sezione [01_concetti_base](01_concetti_base/)

**Storia del ragionamento**:
→ Sezione [02_sviluppo_ipotesi](02_sviluppo_ipotesi/)

**Dettagli test**:
→ Sezione [04_test_osservativi](04_test_osservativi/)

---

## Per tipo di lettore

### 🎓 Studente/Curioso

**Percorso divulgativo**:
1. [Vuoto Cosmico](01_concetti_base/vuoto_cosmico.md)
2. [Fluttuazioni Quantistiche](01_concetti_base/fluttuazioni_quantistiche.md)
3. [Problema Rotazione](01_concetti_base/problema_rotazione_galattica.md)
4. [Verso la GCV](02_sviluppo_ipotesi/verso_la_gcv.md)
5. [Sintesi Finale](05_risultati/sintesi_finale.md)

**Evitare**: formule nella sezione 03 (troppo tecniche)

### 🔬 Ricercatore/Fisico

**Percorso tecnico**:
1. [Formulazione GCV](03_modello_matematico/formulazione_gcv.md)
2. [Test 1](04_test_osservativi/test1_rotazioni_galattiche.md)
3. [Test 2](04_test_osservativi/test2_lensing_gravitazionale.md)
4. [Test 3](04_test_osservativi/test3_cluster_merger.md)
5. [Dati Numerici](05_risultati/dati_numerici.md)
6. [Prossimi Passi](05_risultati/prossimi_passi.md)

**Focus**: verificare metodologia e criteri falsificazione

### 📰 Giornalista/Divulgatore

**Percorso sintetico**:
1. [README.md](README.md)
2. [Problema Rotazione](01_concetti_base/problema_rotazione_galattica.md)
3. [Sintesi Finale](05_risultati/sintesi_finale.md) (sezione "Scenario realistico")

**Tone**: "ipotesi promettente in test" NON "risolta materia oscura"

---

## Struttura delle cartelle

```
teoria_del_vuoto/
│
├── README.md                    # Panoramica generale
├── GLOSSARIO.md                 # Termini e definizioni
├── INDICE.md                    # Questo file
│
├── 01_concetti_base/            # Fondamenti
│   ├── vuoto_cosmico.md
│   ├── fluttuazioni_quantistiche.md
│   ├── materia_oscura.md
│   └── problema_rotazione_galattica.md
│
├── 02_sviluppo_ipotesi/         # Evoluzione pensiero
│   ├── ipotesi_iniziali.md
│   └── verso_la_gcv.md
│
├── 03_modello_matematico/       # Formulazione
│   └── formulazione_gcv.md
│
├── 04_test_osservativi/         # Confronto dati
│   ├── test1_rotazioni_galattiche.md
│   ├── test2_lensing_gravitazionale.md
│   └── test3_cluster_merger.md
│
└── 05_risultati/                # Sintesi e futuro
    ├── sintesi_finale.md
    ├── dati_numerici.md
    └── prossimi_passi.md
```

---

## Riferimenti esterni principali

### Dataset
- **SPARC**: astroweb.case.edu/SPARC
- **SDSS**: sdss.org
- **Planck**: pla.esac.esa.int

### Paper chiave citati
- Mandelbaum+ 2006 (lensing SDSS): arXiv:astro-ph/0605476
- Clowe+ 2006 (Bullet Cluster): arXiv:astro-ph/0608407
- Schombert+ 2020 (SPARC BTFR): arXiv:2001.06251

### Teorie alternative
- MOND: scholarpedia.org/article/MOND
- TeVeS: Bekenstein 2004
- Emergent Gravity: Verlinde 2016

---

## Aggiornamenti

**Versione attuale**: 1.0 (Novembre 2025)

**Ultimo aggiornamento**: Test completati su 27 galassie SPARC, predizioni lensing e cluster formulate.

**Prossimo checkpoint**: Test lensing su stack SDSS (3-6 mesi)

---

## Contatti e contributi

Questo progetto documenta un'ipotesi scientifica in sviluppo.

**Feedback benvenuto** su:
- Errori metodologici
- Suggerimenti per test
- Letteratura rilevante non citata
- Collaborazioni costruttive

**Non benvenuto**:
- Hype prematuro
- Claims non supportati da dati
- Attacchi ad hominem

---

## Licenza e citazioni

**Uso**: 
- Documentazione liberamente consultabile
- Citare se usata in pubblicazioni
- Codici e dati disponibili su richiesta (post peer-review)

**Citazione suggerita**:
```
Teoria GCV (Gravità di Coerenza del Vuoto), 2025
Documentazione disponibile: [link al repository]
```

---

## Note finali

Questa documentazione riflette lo stato dell'ipotesi GCV a Novembre 2025.

**È scienza in progress**, non verità consolidata.

I prossimi test determineranno se la GCV è:
- ✅ Una teoria competitiva con DM
- ⚠️ Un'idea parzialmente corretta
- ❌ Un'ipotesi da rigettare

**Il metodo scientifico richiede onestà**: pubblicheremo i risultati quali che siano.
