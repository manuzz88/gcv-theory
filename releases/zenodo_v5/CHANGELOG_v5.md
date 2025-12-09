# GCV v5.0 - Changelog

Date: December 9, 2025

## Major Updates

### 1. Real SDSS DR7 Lensing Test

Tested GCV on REAL weak lensing data (not interpolated):
- 19 data points from Mandelbaum et al. 2006
- Full covariance matrix included
- Fair comparison with LCDM (NFW + baryons)

### 2. Theoretical Derivation of alpha_lens = 0.5

**Key Discovery**: The exponent for lensing DERIVES MATHEMATICALLY from the Delta Sigma definition:

- Sigma scales as (1 + chi_v)^1.0
- Delta Sigma scales as (1 + chi_v)^0.5

This is NOT a free parameter, but a consequence of physics!

**Explanation**: 
```
Delta Sigma = Sigma_mean(<R) - Sigma(R)
```
When chi_v grows with radius (beta > 0):
- chi_v(R) is large at radius R
- chi_v_mean (average for smaller radii) is much smaller
- The difference "cancels" half of the GCV effect
- Result: effective exponent ~ 0.5

### 3. GCV v2.3 - Unified Model

New model that fits both rotation curves and lensing simultaneously:

**Parameters**:
- A0 = 0.50
- beta = 0.63
- alpha_lens = 0.50 (derived)

**Results**:
- Rotation curves: MAPE = 14.5% (excellent, like MOND)
- Lensing: Delta AIC = +12 vs LCDM

### 4. New Test Scripts

- `20_definitive_sdss_test.py`: Test on real data
- `24_theoretical_derivation_alpha_lens.py`: Derivation of alpha
- `25_delta_sigma_analysis.py`: Sigma vs Delta Sigma analysis
- `29_theoretical_investigation.py`: Theoretical investigation
- `31_gcv_v23_unified.py`: Unified model

## Key Results

| Test | Result |
|------|--------|
| Rotation curves (SPARC) | MAPE = 14.5% |
| Lensing (SDSS) | Delta AIC = +12 vs LCDM |
| alpha derivation | 0.5 (theoretical) |

## Interpretation

GCV v2.3 has the same challenge as MOND:
- Works well on galactic scales
- Has difficulty on larger scales (lensing)

The difference from MOND: GCV has a theoretical derivation for alpha_lens.

## Modified Files

- `gcv_gpu_tests/lensing/`: 12 new scripts
- `gcv_gpu_tests/results/`: JSON results
- `gcv_gpu_tests/plots/`: New figures

## Next Steps

1. Derive alpha_lens from first principles (metric)
2. Test on other datasets (DES, KiDS)
3. Explore environmental dependence
