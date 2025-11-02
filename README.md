Vacuum Coherence Gravity (GCV) Theory
=====================================

Alternative to Dark Matter on Galaxy Scales

Author: Manuel Lazzaro  
Email: manuel.lazzaro@me.com  
Date: November 2025

Overview
--------

This repository contains the code, data, and analysis for the paper:

"Vacuum Coherence Gravity with Growing Susceptibility: A Competitive Alternative to Dark Matter on Galaxy Scales"

GCV proposes that quantum vacuum responds dynamically to matter via scale-dependent susceptibility χᵥ(R,Mb), creating effective gravity amplification without dark matter.

Key Results
-----------

✅ Galaxy Rotation Curves: 10.7% error (SPARC survey, 27 galaxies)  
✅ Cluster Mergers: χ² = 0.90 (3 clusters, τc = 49 Myr)  
✅ MCMC Parameter Optimization: 20,000 samples, R-hat=1.0, perfect convergence
⚠️ Weak Lensing: Preliminary results, simplified ΛCDM comparison

Updated Parameters (MCMC optimized - Nov 2, 2025)
--------------------------------------------------

- a₀ = 1.80×10⁻¹⁰ m/s² (acceleration scale)
- A₀ = 1.16 ± 0.13 (susceptibility amplitude)
- γ = 0.06 ± 0.04 (mass scaling, very weak)
- β = 0.90 ± 0.03 (radial growth, confirmed!)
- τc = 49 ± 8 Myr (vacuum response time)

Repository Structure
--------------------

```
gcv-theory/
├── README.md                 (this file)
├── data/                     (observational data)
│   ├── sparc_rotations.csv  
│   ├── lensing_profiles.csv  
│   └── cluster_offsets.csv
├── notebooks/               (Jupyter notebooks)
│   ├── 01_rotation_curves_test.ipynb
│   ├── 02_weak_lensing_test.ipynb
│   └── 03_cluster_mergers_test.ipynb
├── results/                 (output files)
│   └── gcv_parameters.json
└── plots/                   (figures)
    ├── rotations_fit.png
    ├── lensing_profiles.png
    └── cluster_comparison.png
```

Requirements
------------

- Python 3.9+
- NumPy 1.21+
- SciPy 1.7+
- Matplotlib 3.4+
- Jupyter

Install dependencies:
```bash
pip install numpy scipy matplotlib jupyter
```

Quick Start
-----------

1. Clone repository:
```bash
git clone https://github.com/manuzz88/gcv-theory.git
cd gcv-theory
```

2. Run notebooks in order:
```bash
jupyter notebook notebooks/01_rotation_curves_test.ipynb
```

3. All results are reproducible with fixed random seeds.

Reproducing Results
-------------------

Each notebook corresponds to one test in the paper:

1. Rotation Curves (Test 1):
   - Data: 27 SPARC galaxies
   - Method: v_∞ = (GMa₀)^(1/4)
   - Output: MAPE = 10.7%

2. Weak Lensing (Test 2):
   - Data: Interpolated from Mandelbaum+2006, Leauthaud+2012
   - Method: χᵥ(R,Mb) with growing kernel
   - Output: χ² = 24.4 (preliminary)

3. Cluster Mergers (Test 3):
   - Data: Bullet, El Gordo, MACS J0025
   - Method: τc fit to gas-galaxy offsets
   - Output: τc = 49±8 Myr, χ² = 0.90

Citation
--------

If you use this code or data, please cite:

```
Lazzaro, M. (2025). "Vacuum Coherence Gravity with Growing Susceptibility: 
A Competitive Alternative to Dark Matter on Galaxy Scales." 
arXiv:XXXX.XXXXX [astro-ph.CO]
```

License
-------

MIT License - See LICENSE file

This work is preliminary and provided "as is" for research purposes.

Contact
-------

Manuel Lazzaro  
Email: manuel.lazzaro@me.com  
Phone: +393461587689

Acknowledgments
---------------

- SPARC collaboration for rotation curve data
- SDSS and COSMOS collaborations for lensing data
- AI assistance: Claude (Anthropic) for code development

Version History
---------------

- v1.0 (2025-11-02): Initial release with preliminary results
- Paper status: Preprint submitted to arXiv

Important Notes
---------------

⚠️ PRELIMINARY WORK: The weak lensing results use interpolated data from 
literature and simplified ΛCDM comparison. Full validation with raw catalogs 
and complete baryonic models is needed.

✅ REPRODUCIBLE: All analysis code and data are provided for verification.

🔬 OPEN SCIENCE: Feedback and collaboration welcome!

Last Updated: November 2, 2025
