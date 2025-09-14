import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from ..data.reader_weekly_offdef import read_weekly_file, read_all_weeks


def analyze_multicollinearity(df, correlation_threshold=0.8, vif_threshold=10):
    """
    Comprehensive multicollinearity analysis for NFL weekly data
    
    Parameters:
        df (pd.DataFrame): DataFrame from read_weekly_file
        correlation_threshold (float): High correlation cutoff (default 0.8)
        vif_threshold (float): VIF threshold for multicollinearity (default 10)
    
    Returns:
        tuple: (correlation_matrix, high_correlation_pairs)
    """
    
    # Get only numeric columns, exclude identifiers
    exclude_cols = ['Rk', 'Tm', 'year', 'week', 'G']  # Team identifiers and games
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    analysis_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Calculate correlation matrix
    corr_matrix = df[analysis_cols].corr()
    
    # Visualize correlation heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                fmt='.2f', square=True, cbar_kws={'label': 'Correlation'})
    plt.title('NFL Stats Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Identify high correlations
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > correlation_threshold:
                high_corr_pairs.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j], 
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    print(f"High correlation pairs (|r| > {correlation_threshold}):")
    for pair in high_corr_pairs:
        print(f"  {pair['var1']} ↔ {pair['var2']}: {pair['correlation']:.3f}")
    
    return corr_matrix, high_corr_pairs


def advanced_multicollinearity_tests(df):
    """
    Advanced multicollinearity detection methods beyond correlation
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        tuple: (vif_data, condition_indices, correlation_determinant)
    """
    
    # Prepare data
    exclude_cols = ['Rk', 'Tm', 'year', 'week', 'G']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    analysis_cols = [col for col in numeric_cols if col not in exclude_cols]
    # [print(element) for element in analysis_cols]
    
    # Remove any columns with NaN values for VIF calculation
    clean_df_object = df[analysis_cols].dropna()
    clean_df = clean_df_object.astype(float)
    
    if clean_df.empty:
        print("Warning: No clean data available for analysis")
        return None, None, None
    
    # 1. VARIANCE INFLATION FACTOR (VIF)
    # VIF > 10 indicates high multicollinearity
    # VIF > 5 indicates moderate multicollinearity
    vif_data = pd.DataFrame()
    vif_data["Variable"] = clean_df.columns
    vif_data["VIF"] = [variance_inflation_factor(clean_df.values, i) 
                       for i in range(clean_df.shape[1])]
    vif_data = vif_data.sort_values('VIF', ascending=False)
    
    print("Variance Inflation Factors:")
    print(vif_data)
    print(f"\nHigh VIF (>10): {len(vif_data[vif_data['VIF'] > 10])} variables")
    print(f"Moderate VIF (5-10): {len(vif_data[(vif_data['VIF'] > 5) & (vif_data['VIF'] <= 10)])} variables")
    
    # 2. CONDITION INDEX (Eigenvalue-based detection)
    X_scaled = StandardScaler().fit_transform(clean_df)
    eigenvalues = np.linalg.eigvals(np.dot(X_scaled.T, X_scaled))
    condition_indices = np.sqrt(max(eigenvalues) / eigenvalues)
    
    print(f"\nCondition Indices (>30 indicates multicollinearity):")
    print(f"Max condition index: {max(condition_indices):.2f}")
    severe_multicollinearity = sum(condition_indices > 30)
    print(f"Number of indices > 30: {severe_multicollinearity}")
    
    # 3. DETERMINANT OF CORRELATION MATRIX
    # Close to 0 indicates multicollinearity
    det_corr = np.linalg.det(clean_df.corr())
    print(f"\nCorrelation matrix determinant: {det_corr:.6f}")
    print("(Close to 0 indicates multicollinearity)")
    
    return vif_data, condition_indices, det_corr


def identify_multicollinear_groups(df, correlation_threshold=0.8):
    """
    Identify groups of highly correlated variables
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        correlation_threshold (float): Correlation threshold for grouping (default 0.8)
    
    Returns:
        dict: Dictionary of variable groups
    """
    exclude_cols = ['Rk', 'Tm', 'year', 'week', 'G']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    analysis_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    corr_matrix = df[analysis_cols].corr()
    
    # Find clusters of highly correlated variables
    high_corr_mask = (corr_matrix.abs() > correlation_threshold) & (corr_matrix != 1.0)
    
    # Group related variables
    variable_groups = {}
    processed_vars = set()
    
    for col in corr_matrix.columns:
        if col in processed_vars:
            continue
            
        related_vars = corr_matrix.columns[high_corr_mask[col]].tolist()
        if related_vars:
            # Create group with all related variables
            group_vars = [col] + related_vars
            group_vars = list(set(group_vars))  # Remove duplicates
            
            group_name = f"{col}_group"
            variable_groups[group_name] = group_vars
            
            # Mark all variables in this group as processed
            processed_vars.update(group_vars)
    
    print("Multicollinear variable groups:")
    for group_name, variables in variable_groups.items():
        print(f"  {group_name}: {variables}")
    
    return variable_groups


def full_multicollinearity_analysis(year, week, file_type='def', normalize=False):
    """
    Complete multicollinearity analysis for NFL weekly data
    
    Parameters:
        year (int): Year of data
        week (int): Week number
        file_type (str): Type of file ('def', 'off', 'kicking')
        normalize (bool): Whether to use normalized data
    
    Returns:
        dict: Complete analysis results
    """
    
    # Load data
    df = read_weekly_file(year, week, file_type, normalize=normalize)
    if df is None:
        print(f"Could not load data for {year} week {week} {file_type}")
        return None
    
    print(f"Analyzing {file_type} data for {year} week {week}")
    print(f"Data shape: {df.shape}")
    print(f"Normalized: {normalize}")
    print("-" * 50)
    
    # Run all tests
    try:
        print("=================================")
        print(" Correlation Matrix ")
        corr_matrix, high_corr = analyze_multicollinearity(df)
        print("=================================")
        print(" Variance Inflation Factor (VIF) ")
        vif_data, condition_indices, det_corr = advanced_multicollinearity_tests(df)
        print("=================================")
        print(" Groups Analysis ")
        variable_groups = identify_multicollinear_groups(df)
        
        return {
            'correlation_matrix': corr_matrix,
            'high_correlations': high_corr, 
            'vif': vif_data,
            'condition_indices': condition_indices,
            'determinant': det_corr,
            'variable_groups': variable_groups,
            'data_info': {
                'year': year,
                'week': week,
                'file_type': file_type,
                'normalized': normalize,
                'shape': df.shape
            }
        }
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None


def seasonal_multicollinearity_analysis(year, file_type='def', normalize=False):
    """
    Analyze multicollinearity across an entire season
    
    Parameters:
        year (int): Year of data
        file_type (str): Type of file ('def', 'off', 'kicking')
        normalize (bool): Whether to use normalized data
    
    Returns:
        dict: Season-wide analysis results
    """
    
    # Load season data
    df = read_all_weeks(year, file_type)
    if df is None:
        print(f"Could not load season data for {year} {file_type}")
        return None
    
    print(f"Analyzing {file_type} data for entire {year} season")
    print(f"Data shape: {df.shape}")
    print("-" * 50)
    
    # Run analysis on season data
    try:
        corr_matrix, high_corr = analyze_multicollinearity(df)
        vif_data, condition_indices, det_corr = advanced_multicollinearity_tests(df)
        variable_groups = identify_multicollinear_groups(df)
        
        return {
            'correlation_matrix': corr_matrix,
            'high_correlations': high_corr, 
            'vif': vif_data,
            'condition_indices': condition_indices,
            'determinant': det_corr,
            'variable_groups': variable_groups,
            'data_info': {
                'year': year,
                'file_type': file_type,
                'normalized': normalize,
                'shape': df.shape
            }
        }
    except Exception as e:
        print(f"Error during seasonal analysis: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    print("Running multicollinearity analysis example...")
    
    # Single week analysis
    results = full_multicollinearity_analysis(2022, 6, 'def')
    
    if results:
        print("\nAnalysis completed successfully!")
        print(f"Found {len(results['high_correlations'])} high correlation pairs")
        
        if results['vif'] is not None:
            high_vif = results['vif'][results['vif']['VIF'] > 10]
            print(f"Found {len(high_vif)} variables with high VIF")