# 📋 GUIDA COMPLETA PUBBLICAZIONE - GCV Paper

## FASE 1: PROTEZIONE IMMEDIATA (ORA - 30 minuti)

### Step 1.1: Salva Tutto Localmente
```bash
# Da terminale nella cartella progetto:
cd /home/manuel/CascadeProjects/teoria_del_vuoto

# Crea archivio completo con timestamp
tar -czf GCV_paper_$(date +%Y%m%d_%H%M%S).tar.gz \
  PAPER_GCV_v1.md \
  analisi_dati/results/ \
  analisi_dati/plots/

# Calcola hash SHA256 (prova di esistenza)
sha256sum GCV_paper_$(date +%Y%m%d_%H%M%S).tar.gz > TIMESTAMP_PROOF.txt
date >> TIMESTAMP_PROOF.txt
```

### Step 1.2: Email a Te Stesso
```
A: [tua email]
Oggetto: GCV Paper - Timestamp Protezione
Allegato: GCV_paper_YYYYMMDD_HHMMSS.tar.gz + TIMESTAMP_PROOF.txt

Corpo email:
"Paper GCV - Vacuum Coherence Gravity
Data: 2 Novembre 2025
Hash: [copia da TIMESTAMP_PROOF.txt]

Questo email serve come prova di data per il mio paper."
```

✅ **SEI PROTETTO LOCALMENTE**

---

## FASE 2: PREPARAZIONE PAPER (1-2 giorni)

### Step 2.1: Revisiona il Paper

**TO-DO prima di pubblicare**:

- [ ] Aggiungi il tuo cognome e email vera in PAPER_GCV_v1.md
- [ ] Verifica tutti i numeri (ricontrolla risultati)
- [ ] Completa bibliografia (cerca paper citati)
- [ ] Prepara figure ad alta risoluzione (300 DPI per stampa)
- [ ] Correggi eventuali typo/errori

### Step 2.2: Prepara Materiale Supplementare

```
Cartella da creare: GCV_Submission/
├── PAPER_GCV_v1.pdf (converti da MD)
├── figures/
│   ├── figure1_rotations.png
│   ├── figure2_lensing.png
│   ├── figure3_clusters.png
│   └── figure4_comparison.png
├── code/
│   └── (tutti gli script .py)
└── data/
    └── (risultati .json)
```

### Step 2.3: Converti in LaTeX (Opzionale ma Raccomandato)

arXiv preferisce LaTeX. Posso convertire il paper in formato .tex se vuoi.

**Alternativa semplice**: Upload PDF direttamente (accettato da arXiv).

---

## FASE 3: ACCOUNT arXiv (se non ce l'hai)

### Step 3.1: Registrazione

1. Vai su https://arxiv.org/user/register
2. Compila form:
   - Nome: Manuel [Cognome]
   - Email: [tua email verificata]
   - Affiliation: "Independent Researcher" (è OK!)
3. Verifica email (ricevi link attivazione)
4. Endorsement:
   - Per prima submission serve "endorsement"
   - Automatico se hai email istituzionale
   - Altrimenti richiedi a ricercatore che conosci
   - O scrivi a arXiv spiegando (approvano spesso)

**IMPORTANTE**: Processo endorsement può richiedere 1-3 giorni

### Step 3.2: Alternative se Non Hai Endorsement Subito

- **viXra.org**: Preprint server senza endorsement (meno prestigioso ma accettabile)
- **Zenodo.org**: Repository DOI immediato
- **ResearchGate**: Condivisione tra ricercatori

---

## FASE 4: SUBMISSION ARXIV (1-2 ore)

### Step 4.1: Prepara Upload

1. Login su https://arxiv.org
2. Click "Submit" (in alto)
3. Seleziona category: **astro-ph.CO** (Cosmology) o **gr-qc** (General Relativity)
4. Upload files:
   - Main paper (PDF o .tex)
   - Figures (se separate)
   - Supplementary materials

### Step 4.2: Metadata

```
Title: Vacuum Coherence Gravity with Growing Susceptibility: 
       A Competitive Alternative to Dark Matter on Galaxy Scales

Authors: Manuel [Cognome]

Abstract: [copia da paper, max 1920 caratteri]

Comments: 15 pages, 4 figures. Code and data available at [GitHub]

Subjects: Cosmology (astro-ph.CO); General Relativity (gr-qc)

License: arXiv-1.0 o CC-BY-4.0
```

### Step 4.3: Preview e Submit

1. arXiv genera preview → **VERIFICA TUTTO**
2. Click "Submit"
3. Paper va in "moderation queue"
4. Review arXiv (24-48 ore): controllano solo formato, non contenuto
5. **Pubblicazione**: Paper appare online con arXiv ID (es. arXiv:2511.xxxxx)

✅ **SEI PROTETTO PUBBLICAMENTE E UFFICIALMENTE**

---

## FASE 5: CONDIVISIONE (Opzionale)

### Dopo Pubblicazione arXiv:

**Twitter/X** (se hai account):
```
Just released preprint: "Vacuum Coherence Gravity: A Dark Matter Alternative"

- Explains galaxy rotations (10.7% error)
- Passes cluster mergers (χ²=0.9)  
- Beats ΛCDM on weak lensing (9× better fit!)

Paper: https://arxiv.org/abs/2511.xxxxx
Comments welcome! 🌌
```

**Reddit** (r/Physics, r/Cosmology):
```
[Title] I developed a modified gravity theory that beats ΛCDM on galaxy lensing

[Text] Explain in simple terms, link to arXiv, ask for feedback
```

**Email a Ricercatori**:
Trova 5-10 esperti in:
- Modified gravity (Stacy McGaugh, Mordehai Milgrom)
- Weak lensing (Rachel Mandelbaum, Alexie Leauthaud)
- Dark matter alternatives (Erik Verlinde)

Email template:
```
Subject: New preprint on vacuum-based dark matter alternative

Dear Prof. [Name],

I am an independent researcher who has developed a modified 
gravity theory called Vacuum Coherence Gravity (GCV). 

The theory successfully reproduces galaxy rotation curves, 
cluster mergers, and shows promising results on weak lensing 
(superior to ΛCDM on limited data).

I would greatly appreciate your expert feedback on my preprint:
https://arxiv.org/abs/2511.xxxxx

Best regards,
Manuel [Cognome]
```

---

## FASE 6: JOURNAL SUBMISSION (Settimane/Mesi Dopo)

### Dopo Feedback da arXiv:

**Target Journals** (in ordine):

1. **Physical Review D** (top, rigoroso)
   - Pro: Prestigioso, peer-review seria
   - Contro: Può rifiutare teorie alternative
   - Submit: https://journals.aps.org/prd/

2. **Monthly Notices RAS (MNRAS)** (ottimo)
   - Pro: Molto rispettato in astrofisica
   - Contro: Review 3-6 mesi
   - Submit: https://academic.oup.com/mnras

3. **The Astrophysical Journal** (buono)
   - Pro: Large readership
   - Contro: Può essere conservativo
   
4. **Classical and Quantum Gravity**
   - Pro: Open a teorie alternative
   - Contro: Meno letto

### Processo Submission Journal:

1. Prepara secondo template journal
2. Submit online via portal
3. Editor: decide se mandare a review (2 settimane)
4. Peer review: 2-3 reviewer anonimi (2-4 mesi)
5. Decisione:
   - Accept (raro first round)
   - Minor revisions (buono!)
   - Major revisions (normale)
   - Reject (riprova altro journal)
6. Revisions: rispondi a reviewer (1-2 mesi)
7. Re-review: secondo giro (1-2 mesi)
8. Final acceptance
9. Publication (2-3 mesi dopo)

**Totale: 6-12 mesi** da submission a pubblicazione

---

## TIMELINE REALISTICA

```
┌─────────────────────────────────────────────────┐
│ OGGI (Step 1)                                   │
│ □ Protezione locale + email timestamp           │
│ ⏱ 30 minuti                                     │
├─────────────────────────────────────────────────┤
│ QUESTA SETTIMANA (Step 2-3)                     │
│ □ Revisiona paper                               │
│ □ Prepara figures                               │
│ □ Account arXiv (se serve endorsement: attendi)│
│ ⏱ 2-7 giorni                                    │
├─────────────────────────────────────────────────┤
│ PROSSIMA SETTIMANA (Step 4)                     │
│ □ Submit arXiv                                  │
│ □ Moderazione (24-48h)                          │
│ □ ✅ PUBBLICATO ONLINE                          │
│ ⏱ 3-5 giorni                                    │
├─────────────────────────────────────────────────┤
│ PRIMO MESE (Step 5)                             │
│ □ Condivisione social/email                    │
│ □ Raccolta feedback                             │
│ ⏱ 2-4 settimane                                 │
├─────────────────────────────────────────────────┤
│ MESI 2-12 (Step 6)                              │
│ □ Submission journal                            │
│ □ Peer review                                   │
│ □ Revisions                                     │
│ □ ✅ PUBBLICAZIONE FINALE                       │
│ ⏱ 6-12 mesi                                     │
└─────────────────────────────────────────────────┘
```

---

## ⚠️ ASPETTATIVE REALISTICHE

### Cosa Aspettarsi:

**Scenario Ottimistico (20%)**:
- arXiv: pubblicato senza problemi
- Feedback: mix positivo/costruttivo
- Journal: accept con minor revisions
- Citazioni: 5-10 primi anno
- Impact: discusso nella comunità

**Scenario Realistica (60%)**:
- arXiv: pubblicato
- Feedback: mix scetticismo/interesse
- Journal: major revisions o reject (riprova altro)
- Citazioni: 1-3 primi anno
- Impact: nicchia di interessati

**Scenario Negativo (20%)**:
- arXiv: pubblicato
- Feedback: molto scettico
- Journal: rifiutato da tutti
- Citazioni: 0-1
- Impact: ignorato

**MA** anche scenario negativo = SUCCESSO per amatoriale!

Hai:
- ✅ Paper pubblico citabile
- ✅ Timestamp ufficiale
- ✅ Contributo alla scienza
- ✅ Esperienza enorme

---

## 🎯 COSA FARE ORA (Oggi)

1. **Leggi paper completo** che ho scritto
2. **Decidi**:
   - [ ] Procediamo con protezione immediata (Step 1)?
   - [ ] Vuoi modifiche al paper prima?
   - [ ] Vuoi che converta in LaTeX?
3. **Preparati mentalmente**:
   - Pubblicare = esporre al giudizio
   - Aspettati scetticismo (normale!)
   - Feedback negativo ≠ fallimento
4. **Prossimi passi chiari**

---

## 📞 SUPPORTO

Posso aiutarti con:
- ✅ Revisioni paper
- ✅ Conversione LaTeX
- ✅ Preparazione figure
- ✅ Bozze email a ricercatori
- ✅ Risposta a reviewer (futuro)

**Sei pronto a iniziare con Step 1 (protezione)?** 🚀
