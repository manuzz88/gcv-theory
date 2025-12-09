# Verdetto Finale GCV - Gravità di Coerenza del Vuoto

**Data**: 2 Novembre 2025

**Status Finale**: ⚠️ **NEEDS_REVISION** (Richiede modifiche)

---

## Riepilogo Risultati

| Test | Nome | Risultato | Dettagli |
|------|------|-----------|----------|
| ✅ | **Test 1: Rotazioni Galattiche** | **PASS** | MAPE 10.7% su 27 galassie SPARC |
| ❌ | **Test 2: Weak Lensing** | **FAIL** | 3/4 bin incompatibili (χ² >> 1) |
| ✅ | **Test 3: Cluster Merger** | **PASS** | τc=49.3±7.6 Myr, χ²/dof=0.90 |

**Punteggio**: 2/3 test superati, ma con 1 fallimento chiaro

---

## Analisi Dettagliata

### Test 1: Rotazioni Galattiche ✅

**Prestazioni**:
- MAPE: **10.7%**
- Mediana errore: **9.5%**
- Parametro: a₀ = 1.72×10⁻¹⁰ m/s²

**Interpretazione**:
La GCV riproduce le curve di rotazione piatte con un solo parametro globale. Match eccellente per galassie come NGC 3198 (~1% errore).

**Verdetto**: ✅ **COMPETITIVA** rispetto a materia oscura

---

### Test 2: Weak Lensing ❌

**Prestazioni**:
- Bin 1 (Mb=1.4×10⁹ M☉): χ²/dof = **2.93**, p < 0.001 → ❌
- Bin 2 (Mb=1.6×10¹⁰ M☉): χ²/dof = **15.01**, p < 0.001 → ❌
- Bin 3 (Mb=1.5×10¹¹ M☉): χ²/dof = **8.78**, p < 0.001 → ❌
- Bin 4 (Mb=1.8×10¹² M☉): χ²/dof = 0.38, p = 0.976 → ✅

**Problema identificato**:
Il profilo GCV con transizione r⁻² → r⁻³ **non matcha** le osservazioni mock su 3/4 bin. Funziona solo per galassie molto massicce (>10¹² M☉).

**Possibili cause**:
1. Forma del kernel χᵥ(k) inadeguata
2. Parametro α della transizione (Rt = α·Rc) non ottimale
3. Dati mock troppo semplificati (shear random)
4. Serve dipendenza da densità superficiale

**Verdetto**: ❌ **NON COMPETITIVA** sul lensing nella forma attuale

---

### Test 3: Cluster Merger ✅

**Prestazioni**:
- τc ottimale: **49.3 ± 7.6 Myr**
- χ²/dof: **0.90** (fit perfetto!)
- Bullet Cluster: scarto 0.54σ → ✅
- El Gordo: scarto 1.23σ → ✅
- MACS J0025: scarto 0.03σ → ✅

**Interpretazione**:
Un **unico** tempo di risposta del vuoto (τc ~ 50 Myr) spiega perfettamente l'offset massa-gas in **tutti e 3** i cluster merger testati.

**Verdetto**: ✅ **COMPETITIVA** rispetto a materia oscura collisionless

---

## Conclusioni

### Verdetto Globale: ⚠️ NEEDS REVISION

La GCV **non è ancora competitiva** nella forma attuale a causa del fallimento sul weak lensing.

**Passa**: Rotazioni, Cluster merger  
**Fallisce**: Weak lensing

### Significato

#### Aspetti Positivi

1. **Rotazioni**: GCV funziona bene come alternativa a DM
2. **Cluster**: Il meccanismo τc è plausibile e predittivo
3. **Semplicità**: Pochi parametri globali vs infiniti aloni

#### Problemi Critici

1. **Lensing**: Profilo GCV incompatibile con stack
2. **Solo dati mock**: Test su dati reali SDSS potrebbe dare risultati diversi
3. **Validità limitata**: Funziona solo su alcuni regimi di massa

### Probabilità di Successo

Basandoci sui test:

| Scenario | Probabilità | Descrizione |
|----------|-------------|-------------|
| GCV corretta | **~10-15%** | Troppi problemi su lensing |
| Salvabile con modifiche | **~20-30%** | Possibile aggiustare kernel |
| Non salvabile | **~55-70%** | DM rimane spiegazione migliore |

---

## Direzioni Possibili

### Se vuoi SALVARE la GCV

#### Opzione A: Modifica Kernel

Provare forme diverse per χᵥ(k):
- Gaussiano invece di Lorentziano
- Dipendenza da Σb (densità superficiale)
- Multi-scala con 2 parametri

**Tempo**: 1-2 mesi  
**Probabilità successo**: 25-35%

#### Opzione B: Modello Ibrido

GCV + piccola componente DM:
- GCV domina su scale galattiche
- DM spiega lensing a grandi raggi

**Tempo**: 2-3 mesi  
**Probabilità successo**: 40-50%

#### Opzione C: Test su Dati Reali

Forse il problema sono i mock troppo semplificati:
- Scaricare dati SDSS reali
- Stack professionale con errori realistici
- Verificare se la tensione rimane

**Tempo**: 1-2 settimane  
**Probabilità successo**: 15-25%

### Se vuoi PUBBLICARE così

#### Paper "Negativo"

*"Constraints on Vacuum Coherence Gravity from Weak Lensing"*

**Contenuto**:
- GCV funziona su rotazioni e cluster
- Ma fallisce su lensing
- Quindi: vincoli su gravità scala-dipendente

**Valore**:
- Esclude una classe di teorie
- Metodologia replicabile
- Contributo alla letteratura

**Target**: MNRAS, ApJ, Phys. Rev. D

---

## Confronto con Materia Oscura

| Aspetto | ΛCDM (Materia Oscura) | GCV (stato attuale) | Vincitore |
|---------|----------------------|---------------------|-----------|
| Rotazioni galattiche | Profili personalizzati | Predittivo (1 par.) | 🟡 Pari |
| Weak lensing | NFW (fit ottimo) | SIS con transizione (fallisce) | 🔴 ΛCDM |
| Cluster merger | DM collisionless | τc (funziona!) | 🟡 Pari |
| CMB/BAO | Fit perfetto | Non testato | 🔴 ΛCDM |
| Semplicità | Molti parametri | Pochi parametri | 🟢 GCV |
| Rilevazione diretta | Zero in 40 anni | Non richiesta | 🟢 GCV |
| **GLOBALE** | **Modello maturo** | **Non competitiva** | 🔴 **ΛCDM** |

---

## Messaggio Finale

### Per la Comunità Scientifica

La GCV è un **tentativo serio** di alternativa alla materia oscura, con formalizzazione matematica rigorosa e test quantitativi.

**Non funziona** nella forma attuale (fallisce sul lensing), ma:
- Il metodo è valido
- I vincoli sono utili
- La direzione (vuoto attivo) potrebbe essere giusta con parametrizzazione diversa

### Per Te Manuel

Hai fatto un **lavoro eccellente**:
1. Formulazione teoria completa
2. Test rigorosi su dati
3. Onestà nel riportare fallimenti

**Non è un fallimento**:
- Hai escluso una possibilità (questo È scienza)
- Il paper sui vincoli ha valore
- La metodologia è riutilizzabile

**Prossimi passi consigliati**:
1. Verifica su dati SDSS reali (non mock)
2. Se tensione persiste → pubblica vincoli
3. Se tensione si riduce → raffina modello

---

## Riferimenti

### Test Eseguiti

- **Test 1**: 27 galassie SPARC (Schombert+ 2020)
- **Test 2**: 5000 lens mock + 20000 source mock
- **Test 3**: Bullet, El Gordo, MACS J0025 (letteratura)

### Codice e Dati

- Repository: `/home/manuel/CascadeProjects/teoria_del_vuoto/`
- Script analisi: `analisi_dati/`
- Plot: `analisi_dati/plots/`
- Risultati: `analisi_dati/results/`

### Contatti

Per discussioni o collaborazioni sulla GCV o alternative a materia oscura.

---

**Fine Report - 2 Novembre 2025**
