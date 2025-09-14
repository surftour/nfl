import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ..data.reader_weekly_offdef import read_weekly_file, read_all_weeks


def load_pca_data(year, week=None, file_type='def', normalize=True):
    """
    Load NFL data for PCA analysis
    
    Parameters:
        year (int): Year of data
        week (int, optional): Specific week number. If None, loads all weeks
        file_type (str): Type of file ('def', 'off', 'kicking')
        normalize (bool): Whether to use normalized per-game data
    
    Returns:
        pd.DataFrame: Processed DataFrame ready for PCA, None if error
    """
    
    try:
        if week is not None:
            # Load specific week
            df = read_weekly_file(year, week, file_type, normalize=normalize)
            if df is None:
                print(f"Could not load data for {year} week {week} {file_type}")
                return None
            print(f"Loaded {file_type} data for {year} week {week}")
        else:
            # Load all weeks for the season
            df = read_all_weeks(year, file_type)
            if df is None:
                print(f"Could not load season data for {year} {file_type}")
                return None
            print(f"Loaded {file_type} data for entire {year} season")
        
        print(f"Data shape: {df.shape}")
        print(f"Normalized: {normalize}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def prepare_pca_data(df, exclude_cols=None):
    """
    Prepare DataFrame for PCA analysis by selecting numeric columns
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        exclude_cols (list, optional): Additional columns to exclude
    
    Returns:
        tuple: (feature_matrix, feature_names, scaler) where:
            - feature_matrix: Standardized numpy array for PCA
            - feature_names: List of feature column names
            - scaler: Fitted StandardScaler object
    """
    
    # Default columns to exclude from PCA
    default_exclude = ['Rk', 'Tm', 'year', 'week', 'G']
    if exclude_cols is None:
        exclude_cols = []
    exclude_cols = list(set(default_exclude + exclude_cols))
    
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"Selected {len(feature_cols)} features for PCA")
    print(f"Excluded columns: {exclude_cols}")
    
    # Remove rows with any NaN values in selected features
    clean_df = df[feature_cols].dropna()
    
    if clean_df.empty:
        raise ValueError("No clean data available after removing NaN values")
    
    print(f"Clean data shape: {clean_df.shape}")
    
    # Standardize features (critical for PCA)
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(clean_df)
    
    return feature_matrix, feature_cols, scaler


def build_pca_components(feature_matrix, n_components=None, variance_threshold=0.95):
    """
    Build PCA components from standardized feature matrix
    
    Parameters:
        feature_matrix (np.ndarray): Standardized feature matrix
        n_components (int, optional): Number of components to keep. 
                                    If None, determined by variance_threshold
        variance_threshold (float): Minimum cumulative variance to explain (default 0.95)
    
    Returns:
        tuple: (pca_model, transformed_data, explained_variance_ratio)
    """
    
    # Determine number of components
    if n_components is None:
        # Run PCA with all components first to find optimal number
        pca_full = PCA()
        pca_full.fit(feature_matrix)
        
        # Find number of components needed for variance threshold
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumvar >= variance_threshold) + 1
        
        print(f"Selected {n_components} components to explain {variance_threshold:.1%} of variance")
    
    # Build final PCA model
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(feature_matrix)
    
    print(f"PCA fitted with {n_components} components")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    return pca, transformed_data, pca.explained_variance_ratio_


def analyze_pca_components(pca_model, feature_names, n_top_features=5):
    """
    Analyze and interpret PCA components
    
    Parameters:
        pca_model: Fitted PCA model
        feature_names (list): Names of original features
        n_top_features (int): Number of top contributing features to show per component
    
    Returns:
        pd.DataFrame: Component analysis with top contributing features
    """
    
    components_df = pd.DataFrame(
        pca_model.components_.T,
        columns=[f'PC{i+1}' for i in range(pca_model.n_components_)],
        index=feature_names
    )
    
    print("\nPCA Component Analysis:")
    print("=" * 50)
    
    component_analysis = []
    
    for i, pc in enumerate(components_df.columns):
        print(f"\n{pc} (Explained Variance: {pca_model.explained_variance_ratio_[i]:.3f})")
        print("-" * 30)
        
        # Get top positive and negative loadings
        loadings = components_df[pc].abs().sort_values(ascending=False)
        top_features = loadings.head(n_top_features)
        
        print("Top contributing features:")
        for feature in top_features.index:
            loading = components_df.loc[feature, pc]
            print(f"  {feature}: {loading:.3f}")
        
        component_analysis.append({
            'Component': pc,
            'Explained_Variance': pca_model.explained_variance_ratio_[i],
            'Top_Features': top_features.index.tolist(),
            'Top_Loadings': [components_df.loc[f, pc] for f in top_features.index]
        })
    
    return components_df, component_analysis


def visualize_pca_results(pca_model, transformed_data, feature_names, original_df=None):
    """
    Create visualizations for PCA results
    
    Parameters:
        pca_model: Fitted PCA model
        transformed_data: PCA-transformed data
        feature_names: Original feature names
        original_df: Original DataFrame (optional, for team labels)
    """
    
    # 1. Explained variance plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.bar(range(1, len(pca_model.explained_variance_ratio_) + 1), 
            pca_model.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Component')
    
    # 2. Cumulative explained variance
    plt.subplot(1, 3, 2)
    cumvar = np.cumsum(pca_model.explained_variance_ratio_)
    plt.plot(range(1, len(cumvar) + 1), cumvar, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95%')
    plt.legend()
    
    # 3. PC1 vs PC2 scatter plot
    plt.subplot(1, 3, 3)
    if transformed_data.shape[1] >= 2:
        plt.scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.6)
        plt.xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.3f})')
        plt.title('PC1 vs PC2 Scatter Plot')
        
        # Add team labels if available
        if original_df is not None and 'Tm' in original_df.columns:
            clean_df = original_df.dropna()
            if len(clean_df) == transformed_data.shape[0]:
                for i, team in enumerate(clean_df['Tm']):
                    plt.annotate(team, (transformed_data[i, 0], transformed_data[i, 1]), 
                               fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.show()


def full_pca_analysis(year, week=None, file_type='def', normalize=True, 
                     n_components=None, variance_threshold=0.95):
    """
    Complete PCA analysis workflow for NFL data
    
    Parameters:
        year (int): Year of data
        week (int, optional): Specific week number. If None, uses all weeks
        file_type (str): Type of file ('def', 'off', 'kicking')
        normalize (bool): Whether to use normalized per-game data
        n_components (int, optional): Number of components to keep
        variance_threshold (float): Minimum cumulative variance to explain
    
    Returns:
        dict: Complete PCA analysis results
    """
    
    print(f"Starting PCA analysis for {year} {file_type} data")
    if week is not None:
        print(f"Week: {week}")
    else:
        print("Season: All weeks")
    print(f"Normalized: {normalize}")
    print("=" * 50)
    
    try:
        # Load data
        df = load_pca_data(year, week, file_type, normalize)
        if df is None:
            return None
        
        # Prepare for PCA
        feature_matrix, feature_names, scaler = prepare_pca_data(df)
        
        # Build PCA
        pca_model, transformed_data, explained_var = build_pca_components(
            feature_matrix, n_components, variance_threshold
        )
        
        # Analyze components
        components_df, component_analysis = analyze_pca_components(pca_model, feature_names)
        
        # Visualize results
        visualize_pca_results(pca_model, transformed_data, feature_names, df)
        
        return {
            'pca_model': pca_model,
            'transformed_data': transformed_data,
            'components_matrix': components_df,
            'component_analysis': component_analysis,
            'feature_names': feature_names,
            'scaler': scaler,
            'explained_variance_ratio': explained_var,
            'original_data': df,
            'data_info': {
                'year': year,
                'week': week,
                'file_type': file_type,
                'normalized': normalize,
                'n_components': pca_model.n_components_,
                'total_variance_explained': explained_var.sum()
            }
        }
        
    except Exception as e:
        print(f"Error during PCA analysis: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    print("Running PCA analysis example...")
    
    # Single week analysis
    results = full_pca_analysis(2022, week=6, file_type='def', normalize=True)
    
    if results:
        print("\nPCA Analysis completed successfully!")
        print(f"Total variance explained: {results['data_info']['total_variance_explained']:.3f}")
        print(f"Number of components: {results['data_info']['n_components']}")
        
        # Show component interpretation
        print("\nComponent Interpretation:")
        for comp in results['component_analysis']:
            print(f"{comp['Component']}: {comp['Top_Features'][:3]}")