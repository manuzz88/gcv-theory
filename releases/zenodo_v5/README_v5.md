# Vacuum Coherence Gravity (GCV) Theory - Version 5.0

## Overview

This release contains major updates to the GCV theory, including:
1. Real SDSS DR7 weak lensing tests
2. Theoretical derivation of the lensing exponent alpha = 0.5
3. Unified model GCV v2.3 for rotation curves + lensing

## Key Discovery

**The exponent alpha_lens ~ 0.5 is NOT a free parameter!**

It derives mathematically from the definition of Delta Sigma:
- Sigma scales as (1 + chi_v)^1.0
- Delta Sigma scales as (1 + chi_v)^0.5

This reduces the number of free parameters and increases predictivity.

## Results Summary

### Rotation Curves
- MAPE = 14.5% on SPARC galaxies
- Comparable to MOND performance
- Verdict: EXCELLENT

### Weak Lensing (SDSS DR7)
- Chi2/dof = 2.6
- Delta AIC = +12 vs LCDM
- Verdict: LCDM favored, but GCV competitive

### Unified Model GCV v2.3
Parameters:
- A0 = 0.50
- beta = 0.63  
- alpha_lens = 0.50 (derived)

## Files Included

### Documentation
- `CHANGELOG_v5.md` - This changelog
- `README_v5.md` - This file
- `THEORETICAL_DERIVATION.md` - Mathematical derivation of alpha = 0.5

### Code
- `gcv_gpu_tests/lensing/` - All lensing test scripts
- `gcv_gpu_tests/results/` - JSON results
- `gcv_gpu_tests/plots/` - Generated figures

### Key Scripts
| Script | Description |
|--------|-------------|
| `20_definitive_sdss_test.py` | Real SDSS data test |
| `24_theoretical_derivation_alpha_lens.py` | Alpha derivation |
| `25_delta_sigma_analysis.py` | Sigma vs Delta Sigma |
| `31_gcv_v23_unified.py` | Unified model |

## Citation

If you use this code or data, please cite:

```
Lazzaro, M. (2025). "Vacuum Coherence Gravity v2.3: 
Unified Model for Galaxy Rotation Curves and Weak Lensing"
Zenodo. https://doi.org/10.5281/zenodo.17505642
```

## Contact

Manuel Lazzaro
Email: manuel.lazzaro@me.com

## License

MIT License
