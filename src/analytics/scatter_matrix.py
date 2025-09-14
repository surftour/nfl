#!/usr/bin/env python3
"""
Script to read weekly NFL data and create scatter plot matrix visualization.
Uses read_weekly_file to load data and creates pairwise scatter plots.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.data.reader_weekly_offdef import read_weekly_file


def create_scatter_matrix(data, numeric_columns=None, figsize=(15, 12)):
    """
    Create a scatter plot matrix for numeric columns in the dataset.
    
    Args:
        data: DataFrame with the data to plot
        numeric_columns: List of column names to include (defaults to all numeric)
        figsize: Tuple for figure size
    
    Returns:
        matplotlib figure object
    """
    if numeric_columns is None:
        # Select only numeric columns, excluding identifiers
        exclude_cols = ['Tm', 'Rk', 'year', 'week', 'G']  # Team name, rank, etc.
        numeric_columns = [col for col in data.select_dtypes(include=['float64', 'int64', 'Int64']).columns 
                          if col not in exclude_cols]
    
    # Limit to first 8 columns for readability
    numeric_columns = numeric_columns[:8]
    
    print(f"Creating scatter matrix for columns: {numeric_columns}")
    
    # Create the scatter matrix
    fig = plt.figure(figsize=figsize)
    pd.plotting.scatter_matrix(data[numeric_columns], 
                              alpha=0.7, 
                              diagonal='hist', 
                              figsize=figsize)
    
    plt.suptitle('NFL Team Statistics Scatter Matrix', fontsize=16, y=0.95)
    plt.tight_layout()
    
    return fig


def create_seaborn_pairplot(data, numeric_columns=None):
    """
    Alternative visualization using seaborn's pairplot with better styling.
    
    Args:
        data: DataFrame with the data to plot  
        numeric_columns: List of column names to include
    
    Returns:
        seaborn PairGrid object
    """
    if numeric_columns is None:
        # Select only numeric columns, excluding identifiers
        exclude_cols = ['Tm', 'Rk', 'year', 'week', 'G']
        numeric_columns = [col for col in data.select_dtypes(include=['float64', 'int64', 'Int64']).columns 
                          if col not in exclude_cols]
    
    # Limit to first 6 columns for better readability with seaborn
    numeric_columns = numeric_columns[:6]
    
    print(f"Creating seaborn pairplot for columns: {numeric_columns}")
    
    # Create pairplot
    g = sns.pairplot(data[numeric_columns], 
                     diag_kind='hist',
                     plot_kws={'alpha': 0.7, 's': 30})
    
    g.fig.suptitle('NFL Team Statistics Pair Plot', fontsize=16, y=1.02)
    
    return g


def main():
    """
    Main function to demonstrate scatter matrix creation with NFL weekly data.
    """
    # Example: Read defensive stats for week 6 of 2022
    print("Reading defensive stats for 2022, week 6...")
    def_data = read_weekly_file(2022, 6, 'def', normalize=False)
    
    if def_data is None:
        print("Error: Could not load defensive data")
        return
        
    print(f"Loaded data shape: {def_data.shape}")
    print(f"Available columns: {list(def_data.columns)}")
    print("\nFirst few rows:")
    print(def_data.head())
    
    # Show data types
    print(f"\nData types:")
    print(def_data.dtypes)
    
    # Create matplotlib scatter matrix
    print("\nCreating matplotlib scatter matrix...")
    fig1 = create_scatter_matrix(def_data)
    plt.show()
    
    # Create seaborn pairplot  
    print("\nCreating seaborn pairplot...")
    g = create_seaborn_pairplot(def_data)
    plt.show()
    
    # Example with offensive data
    print("\nReading offensive stats for 2022, week 6...")
    off_data = read_weekly_file(2022, 6, 'off', normalize=False)
    
    if off_data is not None:
        print(f"Offensive data shape: {off_data.shape}")
        
        # Focus on key offensive metrics
        key_off_metrics = ['PF', 'TotYds', 'PassYds', 'RushYds', 'PassTD', 'RushTD']
        available_metrics = [col for col in key_off_metrics if col in off_data.columns]
        
        if available_metrics:
            print(f"\nCreating scatter matrix for key offensive metrics: {available_metrics}")
            fig2 = create_scatter_matrix(off_data, available_metrics, figsize=(12, 10))
            plt.show()


if __name__ == "__main__":
    main()