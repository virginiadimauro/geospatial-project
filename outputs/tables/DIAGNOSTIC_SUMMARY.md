# Spatial Autocorrelation Diagnostics Summary

## Analysis Timeline
1. **OLS Model B** (`scripts/03_ols_price_analysis.py`): Estimated hedonic pricing model with property, host, location, and neighbourhood covariates
2. **Moran's I Tests** (`scripts/04_spatial_autocorr_morans_i.py`): Detected significant listing-level spatial autocorrelation in OLS residuals
3. **LM Diagnostics** (`scripts/05_lm_diagnostic_tests.py`): Determined whether SAR or SEM model is more appropriate

---

## Key Results

### OLS Model B (Baseline)
- **Observations**: 15,641 listings (after price filtering and removing missing values)
- **Regressors**: 140 (property vars + host vars + room type dummies + distance + neighbourhood dummies)
- **R²**: 0.6313
- **Adj R²**: 0.6280
- **Residual std err**: 0.3986

### Moran's I Spatial Autocorrelation Test

**Original Analysis (full sample after price filtering, N=18,940)**:
| Test | k | Moran's I | p-value | Interpretation |
|------|---|-----------|---------|---------------:|
| Listing-level (kNN-8) | 8 | 0.0925 | < 0.001 | **Significant** ✓ |
| Listing-level (kNN-12) | 12 | 0.0729 | < 0.001 | **Significant** ✓ |
| Neighbourhood-level (Queen) | varies | -0.047 | 0.199 | NOT significant |

**Consistency Check (same sample as OLS/LM tests, N=15,641)**:
| Test | k | Moran's I | p-value | Interpretation |
|------|---|-----------|---------|---------------:|
| Listing-level (kNN-8) | 8 | 0.0922 | < 0.001 | **Significant** ✓ |

**Interpretation**: 
- Fine-grained spatial clustering exists at listing level
- Neighbourhood fixed effects do not fully capture spatial dependence
- **✓ Moran's I remains highly significant on OLS regression sample** (confirms spatial autocorrelation is not an artifact of sample selection)

---

## LM Diagnostic Tests for Model Specification

### Test Statistics (k-NN k=8, EPSG:25830, 15,641 obs)

| Test | Statistic | p-value | Decision |
|------|-----------|---------|----------|
| **LM-lag** | 833.07 | < 0.001 | Significant SAR effect |
| LM-error | 585.72 | < 0.001 | Significant SEM effect |
| **RLM-lag** | 412.25 | < 0.001 | Robust SAR effect |
| **RLM-error** | 164.90 | < 0.001 | Robust SEM effect |

### Interpretation

**Decision Rule** (Anselin 1988):
- If RLM-lag < 0.05 AND RLM-error ≥ 0.05 → **Use SAR**
- If RLM-error < 0.05 AND RLM-lag ≥ 0.05 → **Use SEM**
- If both < 0.05 → **Compare empirically** (current case)

**Result**: Both RLM-lag and RLM-error are statistically significant (p < 0.001).

**Preference Indicator**: RLM-lag (412.25) > RLM-error (164.90) → **SAR likely preferred**, but empirical comparison recommended.

---

## 4. Sample Size Reconciliation & Consistency Verification

**Issue**: Three complementary analyses used different sample sizes:
- Script 04 (original Moran's I): N = 18,940 (after price filtering, before covariate validation)
- Script 03 (OLS Model B): N = 15,641 (after complete case analysis)
- Script 05 (LM diagnostics): N = 15,641 (same subset)

**Resolution** (Script 06 - Consistency Check):
- Recalculated Moran's I on exact same subset used by OLS/LM (N=15,641)
- Result: Moran's I = 0.0922, p < 0.001 ✓ **Still highly significant**
- R² = 0.6313 ✓ **Matches Script 05 exactly**

**Implication**: Spatial autocorrelation is robust to sample selection. All three diagnostics (Moran's I, LM-lag, LM-error) now confirmed on identical N=15,641 sample.

---

## 5. Model Specification Recommendation

Based on diagnostic evidence:

1. **OLS is insufficient**: Significant spatial autocorrelation detected in residuals
2. **Specify spatial model**: Either SAR or SEM needed to address spatial dependence
3. **Model choice**: RLM-lag > RLM-error suggests **Spatial Autoregressive (SAR) model** may yield better fit
4. **Next steps** (if needed):
   - Estimate SAR model with spatial lag of log_price
   - Compare SAR vs SEM via information criteria (AIC, BIC)
   - Validate residual spatial autocorrelation in final model

---

## Technical Notes

- **Weight Matrix**: Row-standardized k-NN (k=8, EPSG:25830 metric projection)
- **Spatial Reference**: 
  - Original data CRS: EPSG:4326 (WGS84)
  - Distance calculations (k-NN): EPSG:25830 (UTM Zone 30N)
- **Formula Reference**: Anselin (1988) Spatial Econometrics: Methods and Models
- **Software**: Python, libpysal, geopandas, statsmodels, scipy

---

**Analysis Completed**: [Final date from script execution]  
**Status**: ✓ Diagnostic phase complete; Ready for SAR/SEM specification (if continuing analysis)
