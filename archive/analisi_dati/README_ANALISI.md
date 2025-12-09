# Analisi Dati Raw per Test GCV

## Setup ambiente

Questa cartella contiene gli script per l'analisi completa su dati reali.

## Hardware

- **GPU**: 2x RTX 4000 Ada
- **RAM**: Consigliato 32+ GB
- **Storage**: ~50 GB per dataset completi

## Pipeline

### Test 2: Weak Lensing (SDSS)

**Step**:
1. Download cataloghi SDSS DR12/DR16
2. Selezione galassie lens (con massa stellare)
3. Selezione galassie source (con shear)
4. Binning per massa
5. Stacking e calcolo ΔΣ(R)
6. Confronto con predizioni GCV

### Test 3: Cluster Merger

**Step**:
1. Download dati Chandra (Bullet Cluster)
2. Download mappe lensing HST
3. Analisi offset massa-gas
4. Test τc su 2-3 sistemi

## Tempo stimato

- Setup: 2-4 ore
- Download dati: 4-8 ore (dipende da connessione)
- Processing: 2-6 ore (GPU accelerato)
- Analisi: 2-4 ore

**Totale**: 1-2 giorni
