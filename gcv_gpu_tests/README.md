GCV GPU TESTS - Rigorous Validation
===================================

Goal: Aumentare credibilità da 5% a 35-40% con test rigorosi GPU-accelerated

Hardware: 2x NVIDIA RTX 4000 Ada
Timeline: 2-3 settimane
Status: STARTING - 2 Nov 2025

========================================

STRUTTURA PROGETTO

gcv_gpu_tests/
├── mcmc/                    Test 1: MCMC parameter fitting
├── lensing_raw/            Test 2: Raw SDSS data analysis  
├── lcdm_comparison/        Test 3: Fair ΛCDM comparison
├── simulations/            Test 4: N-body (opzionale)
├── results/                Output JSON/CSV
├── plots/                  Figure professionali
└── data/                   Downloaded catalogs

========================================

ROADMAP

WEEK 1: MCMC Parameter Fitting
- [ ] Setup PyMC3 + JAX GPU
- [ ] Implement GCV likelihood
- [ ] Run 1M iterations MCMC
- [ ] Corner plots + uncertainties
- [ ] Export best-fit parameters
→ Credibilità: 5% → 15%

WEEK 2: Raw Lensing Data
- [ ] Download SDSS DR17 catalog
- [ ] Implement stacking pipeline
- [ ] Bootstrap errors (GPU)
- [ ] Covariance matrix
- [ ] Compare with GCV predictions
→ Credibilità: 15% → 25%

WEEK 3: Fair ΛCDM Comparison
- [ ] NFW + stellar mass + gas
- [ ] Adiabatic contraction
- [ ] Same data as GCV
- [ ] AIC/BIC fair comparison
- [ ] Residuals analysis
→ Credibilità: 25% → 35%

WEEK 4: Paper v2
- [ ] Update paper with new results
- [ ] Professional figures
- [ ] Submit to arXiv (with endorsement)
→ PUBLICATION READY

========================================

DEPENDENCIES

Python 3.9+
numpy>=1.21
scipy>=1.7
matplotlib>=3.4
pymc3>=3.11 (or PyMC>=5.0)
jax>=0.4.0 (GPU support)
jaxlib (with CUDA)
arviz>=0.12 (plotting)
corner>=2.2 (corner plots)
astropy>=5.0 (catalogs)
h5py>=3.0 (data storage)

Install:
pip install pymc jax[cuda] arviz corner astropy h5py

========================================

GPU SETUP

Check GPU:
nvidia-smi

Test JAX GPU:
python -c "import jax; print(jax.devices())"

Expected output:
[cuda(id=0), cuda(id=1)]

========================================

EXPECTED RESULTS

After Week 3:
✅ Parametri robusti con incertezze
✅ Test su dati raw (non interpolati)
✅ Confronto onesto con ΛCDM
✅ Figure professionali per paper
✅ Credibilità 35-40%
✅ Base per collaborazioni

========================================

CONTATTI

Manuel Lazzaro
Email: manuel.lazzaro@me.com
Paper: https://zenodo.org/records/17505642
Code: https://github.com/manuzz88/gcv-theory

========================================

NEXT STEPS

1. Install dependencies (oggi)
2. Test GPU setup (oggi)
3. Start MCMC test (domani)
4. Run Week 1 tasks (prossimi 7 giorni)

LET'S GO! 🚀
