# Using PCA in NFL Data Modeling

Here are the main ways to use PCA in your modeling workflow:

## 1. Dimensionality Reduction for Model Input

```python
from src.analytics.pca import full_pca_analysis
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Get PCA-transformed features
pca_results = full_pca_analysis(2022, file_type='def', normalize=True)
X_pca = pca_results['transformed_data']  # Reduced dimensions
y = target_variable  # e.g., points allowed, wins, etc.

# Train model on PCA features instead of raw features
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2)
model = LinearRegression().fit(X_train, y_train)
```

## 2. Feature Engineering - Use Components as New Features

```python
# Add PCA components as new columns to your dataset
df_with_pca = original_df.copy()
pca_components = pca_results['transformed_data']

# Add first few components as engineered features
df_with_pca['DefensivePC1'] = pca_components[:, 0]  # Overall defensive strength
df_with_pca['DefensivePC2'] = pca_components[:, 1]  # Pass vs run defense
df_with_pca['DefensivePC3'] = pca_components[:, 2]  # Turnover generation

# Use both original and PCA features
X = df_with_pca[['original_features'] + ['DefensivePC1', 'DefensivePC2', 'DefensivePC3']]
```

## 3. Multicollinearity Solution

```python
# When you have multicollinearity issues (high VIF from collinearity.py)
from src.analytics.collinearity import full_multicollinearity_analysis

# Check for multicollinearity
collin_results = full_multicollinearity_analysis(2022, 6, 'def')
high_vif_vars = collin_results['vif'][collin_results['vif']['VIF'] > 10]

if len(high_vif_vars) > 0:
    print("High multicollinearity detected, using PCA...")
    # Use PCA instead of raw correlated features
    pca_results = full_pca_analysis(2022, week=6, file_type='def')
    X_modeling = pca_results['transformed_data']
```

## 4. Combining Offense and Defense PCA

```python
# Create composite team strength metrics
def create_team_pca_features(year, week=None):
    """Create PCA features for both offense and defense"""
    
    # Get offensive PCA
    off_pca = full_pca_analysis(year, week, 'off', normalize=True)
    # Get defensive PCA  
    def_pca = full_pca_analysis(year, week, 'def', normalize=True)
    
    # Combine first few components from each
    team_features = pd.DataFrame({
        'OffensiveStrength': off_pca['transformed_data'][:, 0],
        'OffensiveBalance': off_pca['transformed_data'][:, 1], 
        'DefensiveStrength': def_pca['transformed_data'][:, 0],
        'DefensiveBalance': def_pca['transformed_data'][:, 1],
        'Tm': off_pca['original_data']['Tm']  # Team names for merging
    })
    
    return team_features

# Use in game prediction model
team_ratings = create_team_pca_features(2022, week=6)
```

## 5. Time Series Modeling with PCA

```python
# Track team strength evolution over season using PCA
def track_team_pca_over_season(year, team_id, file_type='def'):
    """Track how team's PCA components change over season"""
    
    team_pca_evolution = []
    
    for week in range(1, 18):  # Regular season weeks
        week_pca = full_pca_analysis(year, week, file_type)
        if week_pca:
            # Find this team's PCA scores
            team_data = week_pca['original_data'][week_pca['original_data']['Tm'] == team_id]
            if not team_data.empty:
                team_idx = team_data.index[0]
                pca_scores = week_pca['transformed_data'][team_idx]
                
                team_pca_evolution.append({
                    'week': week,
                    'PC1': pca_scores[0], 
                    'PC2': pca_scores[1],
                    'PC3': pca_scores[2] if len(pca_scores) > 2 else None
                })
    
    return pd.DataFrame(team_pca_evolution)
```

## 6. Model Interpretation with PCA

```python
# Interpret model predictions through PCA loadings
def interpret_pca_model_prediction(model, pca_results, team_idx):
    """Understand what drives a team's predicted performance"""
    
    # Get team's PCA scores
    team_pca_scores = pca_results['transformed_data'][team_idx]
    
    # Get model coefficients (for linear models)
    coefficients = model.coef_
    
    # Component contributions to prediction
    contributions = team_pca_scores * coefficients
    
    # Map back to original features using component loadings
    components_df = pca_results['components_matrix']
    
    print("Prediction drivers:")
    for i, (pc_score, coef, contrib) in enumerate(zip(team_pca_scores, coefficients, contributions)):
        print(f"PC{i+1}: score={pc_score:.2f}, coef={coef:.2f}, contrib={contrib:.2f}")
        
        # Show top features in this component
        top_features = components_df.iloc[:, i].abs().nlargest(3)
        print(f"  Top features: {list(top_features.index)}")
```

## 7. Preprocessing Pipeline for Production Models

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Create reusable preprocessing pipeline
def create_modeling_pipeline(n_components=None):
    """Create preprocessing pipeline with PCA"""
    
    return Pipeline([
        ('scaler', StandardScaler()),  # Always scale before PCA
        ('pca', PCA(n_components=n_components)),
        ('model', LinearRegression())  # Replace with your model
    ])

# Use pipeline
pipeline = create_modeling_pipeline(n_components=10)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

## Key Benefits for Modeling

The key benefits for modeling:
- **Reduces overfitting** by eliminating redundant features
- **Speeds up training** with fewer dimensions  
- **Handles multicollinearity** automatically
- **Creates interpretable composite metrics** (e.g., "overall offensive strength")
- **Enables cross-team comparisons** on standardized scales