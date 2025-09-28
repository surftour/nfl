# NFL Data Visualization Tools

## Overview

This visualization module provides comprehensive scatter plot analysis tools for NFL game data using the `combine_game_results_with_team_stats` function. The tools allow you to explore relationships between team statistics and game outcomes through interactive visualizations.

## Module Structure

```
src/visualization/
├── __init__.py
├── scatter_analysis.py      # Core scatter plot functionality
├── interactive_explorer.py  # Advanced interactive tools
└── CLAUDE.md               # This documentation
```

## Quick Start

### Basic Scatter Plot

```python
from src.visualization.scatter_analysis import quick_scatter_plot

# Create a scatter plot for Week 1 of 2024
fig = quick_scatter_plot(
    year=2024,
    week=1,
    x_feature='home_PassingYdsPerGame_off',
    y_result='result_home_score'
)
fig.show()
```

### Multi-Week Analysis

```python
from src.visualization.scatter_analysis import multi_week_scatter_plot

# Analyze multiple weeks
fig = multi_week_scatter_plot(
    years=[2023, 2024],
    weeks=[1, 2, 3, 4, 5],
    x_feature='visiting_RushingYdsPerGame_off',
    y_result='result_visiting_score'
)
fig.show()
```

### Interactive Explorer

```python
from src.visualization.interactive_explorer import InteractiveExplorer

# Create interactive explorer
explorer = InteractiveExplorer()
explorer.load_multiple_weeks([2024], [1, 2, 3, 4])

# Find strongest correlations
correlations = explorer.find_strongest_correlations('result_total_points', top_n=10)
print(correlations)

# Create comprehensive feature analysis
fig = explorer.explore_feature_vs_results('home_PassingYdsPerGame_off')
fig.show()
```

## Main Classes

### ScatterAnalyzer

Core class for creating scatter plot analyses.

**Key Methods:**
- `load_data(year, week)` - Load single week data
- `load_multiple_weeks(years, weeks)` - Load multiple weeks
- `create_scatter_plot(x_feature, y_result, **kwargs)` - Create scatter plot
- `create_feature_comparison_plot(features, result)` - Compare multiple features
- `create_correlation_heatmap()` - Generate correlation matrix
- `get_feature_columns()` - List all available features
- `get_result_columns()` - List all available results

### InteractiveExplorer

Advanced interactive analysis tools.

**Key Methods:**
- `explore_feature_vs_results(feature)` - Comprehensive feature analysis
- `compare_home_vs_visiting(stat_pattern, result)` - Home vs away comparison
- `create_team_comparison(teams, features, result)` - Multi-team analysis
- `find_strongest_correlations(result, top_n)` - Find top correlations
- `create_weekly_trend_plot(feature, result, team)` - Weekly trends
- `set_filter(column, value)` - Apply data filters

## Available Data Columns

### Feature Columns (Team Statistics)

Features are prefixed with `home_` or `visiting_` and include:
- **Passing**: `PassingYdsPerGame_off`, `PassingTDPerGame_off`, `PassingIntPerGame_off`
- **Rushing**: `RushingYdsPerGame_off`, `RushingTDPerGame_off`
- **Total**: `TotYdsPlusTOYPerP_off`, `TotYdsPlusTOYPerP_def`
- **Penalties**: `PenaltiesYdsPerGame_off`, `PenaltiesYdsPerGame_def`
- **Defense**: Various defensive statistics with `_def` suffix

### Result Columns (Game Outcomes)

- `result_visiting_score` - Visiting team's final score
- `result_home_score` - Home team's final score
- `result_visiting_win` - Binary: 1 if visiting team won
- `result_home_win` - Binary: 1 if home team won
- `result_total_points` - Sum of both teams' scores
- `result_point_differential` - Home score minus visiting score

## Usage Examples

### 1. Basic Feature vs Result Analysis

```python
from src.visualization.scatter_analysis import ScatterAnalyzer

# Initialize analyzer
analyzer = ScatterAnalyzer()
analyzer.load_data(2024, 5)  # Week 5 of 2024

# Get available columns
features = analyzer.get_feature_columns()
results = analyzer.get_result_columns()

print("Available features:", features[:5])  # Show first 5
print("Available results:", results)

# Create scatter plot
fig = analyzer.create_scatter_plot(
    x_feature='home_PassingYdsPerGame_off',
    y_result='result_home_score',
    show_regression=True,
    show_correlation=True
)
fig.show()
```

### 2. Correlation Analysis

```python
# Create correlation heatmap
fig = analyzer.create_correlation_heatmap(
    features=['home_PassingYdsPerGame_off', 'home_RushingYdsPerGame_off', 'home_TotYdsPlusTOYPerP_off'],
    results=['result_home_score', 'result_total_points']
)
fig.show()
```

### 3. Multi-Feature Comparison

```python
# Compare multiple features against one result
features_to_compare = [
    'home_PassingYdsPerGame_off',
    'home_RushingYdsPerGame_off',
    'home_TotYdsPlusTOYPerP_off'
]

fig = analyzer.create_feature_comparison_plot(
    features=features_to_compare,
    result='result_home_score'
)
fig.show()
```

### 4. Interactive Analysis with Filters

```python
from src.visualization.interactive_explorer import InteractiveExplorer

# Create explorer and load multiple weeks
explorer = InteractiveExplorer()
explorer.load_multiple_weeks([2024], list(range(1, 6)))  # Weeks 1-5

# Apply filters
explorer.set_filter('year', 2024)
explorer.set_filter('result_home_team', 'Philadelphia Eagles')

# Find strongest correlations for home score
correlations = explorer.find_strongest_correlations('result_home_score', top_n=5)
print("Top 5 correlations with home score:")
print(correlations[['feature', 'correlation', 'n_observations']])

# Create comprehensive analysis
fig = explorer.explore_feature_vs_results('home_PassingYdsPerGame_off')
fig.show()
```

### 5. Team-Specific Analysis

```python
# Compare specific teams
teams_to_compare = ['Dallas Cowboys', 'Philadelphia Eagles', 'New York Giants']

fig = explorer.create_team_comparison(
    teams=teams_to_compare,
    features=['PassingYdsPerGame_off', 'RushingYdsPerGame_off'],
    result='result_total_points'
)
fig.show()
```

### 6. Home vs Visiting Team Analysis

```python
# Compare home vs visiting team performance
fig = explorer.compare_home_vs_visiting(
    stat_pattern='PassingYdsPerGame_off',
    result='result_total_points'
)
fig.show()
```

### 7. Weekly Trends

```python
# Analyze weekly trends for a specific team
fig = explorer.create_weekly_trend_plot(
    feature='home_PassingYdsPerGame_off',
    result='result_home_score',
    team='Philadelphia Eagles'
)
fig.show()
```

## Statistical Features

All visualizations include:
- **Regression lines** - Linear trend lines
- **Correlation coefficients** - Pearson correlation with p-values
- **Sample sizes** - Number of observations
- **Confidence intervals** - Statistical uncertainty bounds

## Customization Options

### Plot Customization

```python
fig = analyzer.create_scatter_plot(
    x_feature='home_PassingYdsPerGame_off',
    y_result='result_home_score',
    color_by='result_home_team',  # Color by team
    figsize=(12, 8),              # Custom size
    alpha=0.6,                    # Point transparency
    show_regression=True,         # Show trend line
    show_correlation=True         # Show correlation stats
)
```

### Filtering

```python
# Filter by specific conditions
fig = analyzer.create_scatter_plot(
    x_feature='home_PassingYdsPerGame_off',
    y_result='result_home_score',
    filter_dict={
        'year': 2024,
        'week': 5,
        'result_home_team': 'Philadelphia Eagles'
    }
)
```

## Performance Tips

1. **Load specific weeks** rather than entire seasons for faster analysis
2. **Use filters** to focus on subsets of interest
3. **Cache results** when exploring multiple visualizations
4. **Batch operations** when creating multiple plots

## Error Handling

The tools include robust error handling for:
- Missing data files
- Invalid column names
- Empty datasets after filtering
- Correlation calculations with insufficient data

## Dependencies

- `pandas>=2.0.0` - Data manipulation
- `matplotlib>=3.5.0` - Plotting
- `seaborn>=0.11.0` - Enhanced visualizations
- `numpy>=1.20.0` - Numerical operations
- `scipy>=1.7.0` - Statistical functions

## Integration with Existing Tools

The visualization tools integrate seamlessly with:
- `src.analytics.game_team_stats.combine_game_results_with_team_stats()`
- `models/data.py` data collection functions
- `src.analytics.collinearity` multicollinearity analysis
- Jupyter notebooks for interactive exploration

## Example Notebook Workflow

```python
# Complete analysis workflow
from src.visualization.interactive_explorer import InteractiveExplorer

# 1. Load data
explorer = InteractiveExplorer()
explorer.load_multiple_weeks([2023, 2024], list(range(1, 10)))

# 2. Get data summary
summary = explorer.get_data_summary()
print(f"Analyzing {summary['total_games']} games")

# 3. Find top correlations
for result in ['result_home_score', 'result_total_points']:
    print(f"\nTop correlations with {result}:")
    corr_df = explorer.find_strongest_correlations(result, top_n=3)
    for _, row in corr_df.iterrows():
        print(f"  {row['feature']}: r = {row['correlation']:.3f}")

# 4. Create visualizations
results = explorer.analyzer.get_result_columns()
features = explorer.analyzer.get_feature_columns()

# Feature vs all results
fig1 = explorer.explore_feature_vs_results(features[0])

# Correlation heatmap
fig2 = explorer.analyzer.create_correlation_heatmap(
    features=features[:10],
    results=results[:4]
)

# Show plots
fig1.show()
fig2.show()
```