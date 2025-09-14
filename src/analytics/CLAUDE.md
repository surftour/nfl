# Analytics Module Documentation

## Multicollinearity Analysis

The `collinearity.py` module provides comprehensive multicollinearity detection for NFL weekly data from the `reader_weekly_offdef` module.

### Functions

#### `analyze_multicollinearity(df, correlation_threshold=0.8, vif_threshold=10)`

Primary multicollinearity analysis with correlation matrix visualization.

**Parameters:**
- `df` (pd.DataFrame): DataFrame from read_weekly_file
- `correlation_threshold` (float): High correlation cutoff (default 0.8)  
- `vif_threshold` (float): VIF threshold for multicollinearity (default 10)

**Returns:**
- `tuple`: (correlation_matrix, high_correlation_pairs)

**Features:**
- Generates correlation heatmap visualization
- Excludes identifier columns ('Rk', 'Tm', 'year', 'week', 'G')
- Identifies variable pairs exceeding correlation threshold

#### `advanced_multicollinearity_tests(df)`

Advanced statistical tests for multicollinearity detection.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame

**Returns:**
- `tuple`: (vif_data, condition_indices, correlation_determinant)

**Methods:**
1. **Variance Inflation Factor (VIF)**: Quantifies variance increase due to collinearity
2. **Condition Index**: Eigenvalue-based detection method
3. **Correlation Matrix Determinant**: Overall multicollinearity indicator

#### `identify_multicollinear_groups(df, correlation_threshold=0.8)`

Groups highly correlated variables for easier interpretation.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `correlation_threshold` (float): Correlation threshold for grouping (default 0.8)

**Returns:**
- `dict`: Dictionary of variable groups

#### `full_multicollinearity_analysis(year, week, file_type='def', normalize=False)`

Complete analysis workflow for single week data.

**Parameters:**
- `year` (int): Year of data
- `week` (int): Week number
- `file_type` (str): Type of file ('def', 'off', 'kicking')
- `normalize` (bool): Whether to use normalized data

#### `seasonal_multicollinearity_analysis(year, file_type='def', normalize=False)`

Analysis across entire season using `read_all_weeks()`.

## Multicollinearity Detection Thresholds

### Correlation-Based Thresholds

| Threshold | Interpretation | Action |
|-----------|----------------|--------|
| \|r\| > 0.9 | Severe multicollinearity | Consider removing one variable |
| \|r\| > 0.8 | High multicollinearity | Investigate relationship |
| \|r\| > 0.7 | Moderate multicollinearity | Monitor in models |
| \|r\| < 0.7 | Acceptable | No immediate action needed |

### Variance Inflation Factor (VIF) Thresholds

| VIF Range | Interpretation | Recommendation |
|-----------|----------------|----------------|
| VIF > 10 | High multicollinearity | Remove or combine variables |
| 5 < VIF ≤ 10 | Moderate multicollinearity | Consider transformation |
| VIF ≤ 5 | Low multicollinearity | Acceptable for modeling |

**Formula:** `VIF = 1 / (1 - R²)` where R² is from regressing variable on all others

### Condition Index Thresholds

| Condition Index | Interpretation |
|-----------------|----------------|
| CI > 30 | Severe multicollinearity present |
| 15 < CI ≤ 30 | Moderate multicollinearity |
| CI ≤ 15 | No serious multicollinearity |

### Correlation Matrix Determinant

| Determinant Value | Interpretation |
|-------------------|----------------|
| Close to 0 | Perfect multicollinearity |
| 0 < det < 0.1 | High multicollinearity |
| det ≥ 0.1 | Acceptable multicollinearity |

## Expected NFL Data Correlations

Based on NFL statistics structure, expect high correlations between:

### Offensive Statistics
- **Passing**: `PassingCmp` ↔ `PassingAtt` ↔ `PassingYds`
- **Rushing**: `RushingAtt` ↔ `RushingYds` 
- **Total Production**: `Yds` ↔ `PassingYds` + `RushingYds`
- **Scoring**: `PF` ↔ `PassingTD` + `RushingTD`

### Defensive Statistics  
- **Yards Allowed**: `Yds` ↔ `TotYdsPlusTOPly` ↔ `PassingYds` + `RushingYds`
- **Turnovers**: `FL` ↔ `PassingInt` ↔ `TO%`
- **First Downs**: `1stD` ↔ `Passing1stD` + `Rushing1stD`

### Universal Correlations
- **Penalties**: `PenaltiesPen` ↔ `PenaltiesYds`
- **Games**: All cumulative stats correlate with `G` (games played)

## Usage Examples

```python
from src.analytics.collinearity import (
    full_multicollinearity_analysis,
    seasonal_multicollinearity_analysis
)

# Single week analysis
results = full_multicollinearity_analysis(2022, 6, 'def', normalize=False)

# Season analysis  
season_results = seasonal_multicollinearity_analysis(2022, 'off')

# Access results
if results:
    high_corr_pairs = results['high_correlations']
    vif_data = results['vif'] 
    variable_groups = results['variable_groups']
```

## Dependencies

- `pandas>=2.0.0`: Data manipulation
- `numpy>=1.20.0`: Numerical computations  
- `matplotlib>=3.5.0`: Correlation heatmap visualization
- `seaborn>=0.11.0`: Enhanced plotting
- `statsmodels>=0.13.0`: VIF calculations
- `scikit-learn>=1.0.0`: Data preprocessing

## Notes

- All functions automatically exclude non-statistical columns (team names, identifiers)
- VIF calculations require complete data (no NaN values)
- Correlation matrices handle missing data through pairwise deletion
- Seasonal analysis provides more robust correlation estimates than single-week data
- Use `normalize=True` for per-game stats, `normalize=False` for cumulative stats