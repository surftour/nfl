"""
NFL Data Scatter Plot Analysis Tools

Provides comprehensive scatter plot visualization for analyzing relationships between
team statistics and game outcomes using data from combine_game_results_with_team_stats.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from scipy import stats
import warnings

# Import the data function
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from analytics.game_team_stats import combine_game_results_with_team_stats


class ScatterAnalyzer:
    """
    Main class for creating scatter plot analyses of NFL game data.
    """

    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize the scatter analyzer.

        Args:
            data (pd.DataFrame, optional): Pre-loaded combined game data.
                If None, use load_data() to load specific weeks.
        """
        self.data = data
        self._feature_cache = None
        self._result_cache = None

    def load_data(self, year: int, week: int) -> None:
        """
        Load data for a specific year and week.

        Args:
            year (int): Season year
            week (int): Week number
        """
        self.data = combine_game_results_with_team_stats(year, week)
        self._clear_cache()

    def load_multiple_weeks(self, years: List[int], weeks: List[int]) -> None:
        """
        Load data from multiple years and weeks.

        Args:
            years (List[int]): List of years
            weeks (List[int]): List of weeks
        """
        all_data = []
        for year in years:
            for week in weeks:
                try:
                    df = combine_game_results_with_team_stats(year, week)
                    if df is not None and not df.empty:
                        all_data.append(df)
                except Exception as e:
                    warnings.warn(f"Could not load data for {year} week {week}: {e}")

        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            self._clear_cache()
        else:
            raise ValueError("No data could be loaded")

    def _clear_cache(self) -> None:
        """Clear cached feature and result lists."""
        self._feature_cache = None
        self._result_cache = None

    def get_feature_columns(self) -> List[str]:
        """
        Get all available feature columns (team statistics).

        Returns:
            List[str]: List of feature column names
        """
        if self._feature_cache is None:
            if self.data is None:
                raise ValueError("No data loaded. Use load_data() first.")

            # Get all columns that start with 'visiting_' or 'home_' and are numeric
            feature_cols = []
            for col in self.data.columns:
                if (col.startswith('visiting_') or col.startswith('home_')) and self.data[col].dtype in ['float64', 'int64']:
                    feature_cols.append(col)

            self._feature_cache = sorted(feature_cols)

        return self._feature_cache

    def get_result_columns(self) -> List[str]:
        """
        Get all available result columns (game outcomes).

        Returns:
            List[str]: List of result column names
        """
        if self._result_cache is None:
            if self.data is None:
                raise ValueError("No data loaded. Use load_data() first.")

            result_cols = [col for col in self.data.columns if col.startswith('result_')]
            self._result_cache = sorted(result_cols)

        return self._result_cache

    def create_scatter_plot(self,
                          x_feature: str,
                          y_result: str,
                          color_by: Optional[str] = None,
                          filter_dict: Optional[Dict[str, Any]] = None,
                          show_regression: bool = True,
                          show_correlation: bool = True,
                          figsize: Tuple[int, int] = (10, 6),
                          alpha: float = 0.7) -> plt.Figure:
        """
        Create a scatter plot of a feature vs a result variable.

        Args:
            x_feature (str): Feature column name for x-axis
            y_result (str): Result column name for y-axis
            color_by (str, optional): Column to color points by
            filter_dict (dict, optional): Dictionary of column:value pairs to filter data
            show_regression (bool): Whether to show regression line
            show_correlation (bool): Whether to show correlation coefficient
            figsize (tuple): Figure size (width, height)
            alpha (float): Point transparency (0-1)

        Returns:
            matplotlib.Figure: The created figure
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")

        # Validate columns
        if x_feature not in self.data.columns:
            raise ValueError(f"Feature '{x_feature}' not found in data")
        if y_result not in self.data.columns:
            raise ValueError(f"Result '{y_result}' not found in data")

        # Filter data if requested
        plot_data = self.data.copy()
        if filter_dict:
            for col, val in filter_dict.items():
                if col in plot_data.columns:
                    plot_data = plot_data[plot_data[col] == val]

        # Remove rows with missing values
        plot_data = plot_data.dropna(subset=[x_feature, y_result])

        if plot_data.empty:
            raise ValueError("No data remaining after filtering and removing missing values")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create scatter plot
        if color_by and color_by in plot_data.columns:
            # Use seaborn for colored scatter
            sns.scatterplot(data=plot_data, x=x_feature, y=y_result,
                          hue=color_by, alpha=alpha, ax=ax)
        else:
            ax.scatter(plot_data[x_feature], plot_data[y_result], alpha=alpha)

        # Add regression line
        if show_regression and len(plot_data) > 1:
            x_vals = plot_data[x_feature].values
            y_vals = plot_data[y_result].values
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            ax.plot(x_vals, p(x_vals), "r--", alpha=0.8, linewidth=2)

        # Calculate and display correlation
        correlation = None
        if show_correlation and len(plot_data) > 1:
            correlation, p_value = stats.pearsonr(plot_data[x_feature], plot_data[y_result])
            ax.text(0.05, 0.95, f'r = {correlation:.3f}\np = {p_value:.3f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Formatting
        ax.set_xlabel(self._format_column_name(x_feature))
        ax.set_ylabel(self._format_column_name(y_result))
        ax.set_title(f'{self._format_column_name(y_result)} vs {self._format_column_name(x_feature)}')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_feature_comparison_plot(self,
                                     features: List[str],
                                     result: str,
                                     figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create a grid of scatter plots comparing multiple features against one result.

        Args:
            features (List[str]): List of feature column names
            result (str): Result column name
            figsize (tuple): Figure size (width, height)

        Returns:
            matplotlib.Figure: The created figure
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")

        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, feature in enumerate(features):
            if i < len(axes):
                ax = axes[i]

                # Remove missing values
                plot_data = self.data.dropna(subset=[feature, result])

                if not plot_data.empty:
                    ax.scatter(plot_data[feature], plot_data[result], alpha=0.6)

                    # Add regression line
                    x_vals = plot_data[feature].values
                    y_vals = plot_data[result].values
                    if len(x_vals) > 1:
                        z = np.polyfit(x_vals, y_vals, 1)
                        p = np.poly1d(z)
                        ax.plot(x_vals, p(x_vals), "r--", alpha=0.8)

                        # Add correlation
                        correlation, _ = stats.pearsonr(x_vals, y_vals)
                        ax.text(0.05, 0.95, f'r = {correlation:.3f}',
                               transform=ax.transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                ax.set_xlabel(self._format_column_name(feature))
                ax.set_ylabel(self._format_column_name(result))
                ax.set_title(f'{self._format_column_name(feature)}')
                ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f'Feature Analysis: {self._format_column_name(result)}', fontsize=16)
        plt.tight_layout()
        return fig

    def create_correlation_heatmap(self,
                                 features: Optional[List[str]] = None,
                                 results: Optional[List[str]] = None,
                                 figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create a correlation heatmap between features and results.

        Args:
            features (List[str], optional): List of features. If None, uses all features.
            results (List[str], optional): List of results. If None, uses all results.
            figsize (tuple): Figure size (width, height)

        Returns:
            matplotlib.Figure: The created figure
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")

        if features is None:
            features = self.get_feature_columns()
        if results is None:
            results = self.get_result_columns()

        # Select only the columns we want and remove missing values
        cols_to_use = features + results
        correlation_data = self.data[cols_to_use].dropna()

        # Calculate correlation matrix
        corr_matrix = correlation_data.corr()

        # Extract correlations between features and results
        feature_result_corr = corr_matrix.loc[features, results]

        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(feature_result_corr, annot=True, cmap='RdBu_r', center=0,
                   fmt='.3f', ax=ax, cbar_kws={'label': 'Correlation Coefficient'})

        ax.set_title('Feature-Result Correlation Matrix')
        ax.set_xlabel('Results')
        ax.set_ylabel('Features')

        # Format labels
        formatted_results = [self._format_column_name(col) for col in results]
        formatted_features = [self._format_column_name(col) for col in features]
        ax.set_xticklabels(formatted_results, rotation=45, ha='right')
        ax.set_yticklabels(formatted_features, rotation=0)

        plt.tight_layout()
        return fig

    def _format_column_name(self, col_name: str) -> str:
        """
        Format column names for display in plots.

        Args:
            col_name (str): Original column name

        Returns:
            str: Formatted column name
        """
        # Remove prefixes and clean up names
        formatted = col_name.replace('visiting_', '').replace('home_', '').replace('result_', '')
        formatted = formatted.replace('_', ' ').title()
        return formatted

    def get_summary_statistics(self, feature: str, result: str) -> Dict[str, float]:
        """
        Get summary statistics for a feature-result pair.

        Args:
            feature (str): Feature column name
            result (str): Result column name

        Returns:
            Dict[str, float]: Dictionary of statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")

        clean_data = self.data.dropna(subset=[feature, result])

        if clean_data.empty:
            return {}

        correlation, p_value = stats.pearsonr(clean_data[feature], clean_data[result])

        return {
            'correlation': correlation,
            'p_value': p_value,
            'n_observations': len(clean_data),
            'feature_mean': clean_data[feature].mean(),
            'feature_std': clean_data[feature].std(),
            'result_mean': clean_data[result].mean(),
            'result_std': clean_data[result].std()
        }


def quick_scatter_plot(year: int,
                      week: int,
                      x_feature: str,
                      y_result: str,
                      **kwargs) -> plt.Figure:
    """
    Quick function to create a scatter plot for a single week.

    Args:
        year (int): Season year
        week (int): Week number
        x_feature (str): Feature column name for x-axis
        y_result (str): Result column name for y-axis
        **kwargs: Additional arguments passed to create_scatter_plot()

    Returns:
        matplotlib.Figure: The created figure
    """
    analyzer = ScatterAnalyzer()
    analyzer.load_data(year, week)
    return analyzer.create_scatter_plot(x_feature, y_result, **kwargs)


def multi_week_scatter_plot(years: List[int],
                           weeks: List[int],
                           x_feature: str,
                           y_result: str,
                           **kwargs) -> plt.Figure:
    """
    Quick function to create a scatter plot for multiple weeks.

    Args:
        years (List[int]): List of years
        weeks (List[int]): List of weeks
        x_feature (str): Feature column name for x-axis
        y_result (str): Result column name for y-axis
        **kwargs: Additional arguments passed to create_scatter_plot()

    Returns:
        matplotlib.Figure: The created figure
    """
    analyzer = ScatterAnalyzer()
    analyzer.load_multiple_weeks(years, weeks)
    return analyzer.create_scatter_plot(x_feature, y_result, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("Loading NFL data for analysis...")

    # Create analyzer and load data
    analyzer = ScatterAnalyzer()
    analyzer.load_data(2024, 1)  # Load Week 1 of 2024

    print("Available features:")
    features = analyzer.get_feature_columns()
    for i, feature in enumerate(features[:10]):  # Show first 10
        print(f"  {i+1}. {feature}")

    print("\nAvailable results:")
    results = analyzer.get_result_columns()
    for i, result in enumerate(results):
        print(f"  {i+1}. {result}")

    if features and results:
        # Create example scatter plot
        fig = analyzer.create_scatter_plot(features[0], results[0])
        plt.show()