"""
Interactive NFL Data Explorer

Provides interactive tools for exploring relationships in NFL game data with
advanced filtering, comparison capabilities, and statistical analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
import warnings
from itertools import combinations

# Import the scatter analyzer
from .scatter_analysis import ScatterAnalyzer


class InteractiveExplorer:
    """
    Interactive explorer for NFL data with advanced filtering and comparison tools.
    """

    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize the interactive explorer.

        Args:
            data (pd.DataFrame, optional): Pre-loaded combined game data
        """
        self.analyzer = ScatterAnalyzer(data)
        self.data = data
        self.current_filters = {}

    def load_data(self, year: int, week: int) -> None:
        """Load data for analysis."""
        self.analyzer.load_data(year, week)
        self.data = self.analyzer.data
        self.current_filters = {}

    def load_multiple_weeks(self, years: List[int], weeks: List[int]) -> None:
        """Load data from multiple weeks."""
        self.analyzer.load_multiple_weeks(years, weeks)
        self.data = self.analyzer.data
        self.current_filters = {}

    def set_filter(self, column: str, value: Any) -> None:
        """
        Set a filter for the current analysis.

        Args:
            column (str): Column name to filter on
            value (Any): Value to filter by
        """
        self.current_filters[column] = value

    def clear_filters(self) -> None:
        """Clear all current filters."""
        self.current_filters = {}

    def list_unique_values(self, column: str) -> List[Any]:
        """
        List unique values in a column for filtering.

        Args:
            column (str): Column name

        Returns:
            List[Any]: Unique values in the column
        """
        if self.data is None:
            raise ValueError("No data loaded")

        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found")

        return sorted(self.data[column].dropna().unique())

    def explore_feature_vs_results(self,
                                 feature: str,
                                 figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
        """
        Create a comprehensive analysis of one feature against all result variables.

        Args:
            feature (str): Feature to analyze
            figsize (tuple): Figure size

        Returns:
            matplotlib.Figure: The created figure
        """
        results = self.analyzer.get_result_columns()

        # Filter out binary win columns for cleaner visualization
        numeric_results = [r for r in results if not r.endswith('_win')]

        n_results = len(numeric_results)
        n_cols = 2
        n_rows = (n_results + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_results == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, result in enumerate(numeric_results):
            if i < len(axes):
                ax = axes[i]

                # Get filtered data
                plot_data = self._apply_filters()
                plot_data = plot_data.dropna(subset=[feature, result])

                if not plot_data.empty:
                    ax.scatter(plot_data[feature], plot_data[result], alpha=0.6)

                    # Add regression line and correlation
                    x_vals = plot_data[feature].values
                    y_vals = plot_data[result].values

                    if len(x_vals) > 1:
                        z = np.polyfit(x_vals, y_vals, 1)
                        p = np.poly1d(z)
                        ax.plot(x_vals, p(x_vals), "r--", alpha=0.8)

                        correlation = np.corrcoef(x_vals, y_vals)[0, 1]
                        ax.text(0.05, 0.95, f'r = {correlation:.3f}\nn = {len(x_vals)}',
                               transform=ax.transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                ax.set_xlabel(self._format_name(feature))
                ax.set_ylabel(self._format_name(result))
                ax.set_title(f'{self._format_name(result)}')
                ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_results, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f'Feature Analysis: {self._format_name(feature)}', fontsize=16)
        plt.tight_layout()
        return fig

    def compare_home_vs_visiting(self,
                               stat_pattern: str,
                               result: str,
                               figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """
        Compare home vs visiting team statistics for a given pattern.

        Args:
            stat_pattern (str): Pattern to match (e.g., 'PassingYdsPerGame')
            result (str): Result variable to analyze
            figsize (tuple): Figure size

        Returns:
            matplotlib.Figure: The created figure
        """
        # Find matching home and visiting columns
        home_col = None
        visiting_col = None

        for col in self.data.columns:
            if f'home_{stat_pattern}' in col:
                home_col = col
            elif f'visiting_{stat_pattern}' in col:
                visiting_col = col

        if not home_col or not visiting_col:
            available_patterns = set()
            for col in self.data.columns:
                if col.startswith('home_') or col.startswith('visiting_'):
                    pattern = col.replace('home_', '').replace('visiting_', '')
                    available_patterns.add(pattern)

            raise ValueError(f"Could not find columns for pattern '{stat_pattern}'. "
                           f"Available patterns: {sorted(available_patterns)}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        plot_data = self._apply_filters()

        # Home team analysis
        home_data = plot_data.dropna(subset=[home_col, result])
        if not home_data.empty:
            ax1.scatter(home_data[home_col], home_data[result], alpha=0.6, color='blue')
            self._add_regression_line(ax1, home_data[home_col], home_data[result])

        ax1.set_xlabel(self._format_name(home_col))
        ax1.set_ylabel(self._format_name(result))
        ax1.set_title('Home Team')
        ax1.grid(True, alpha=0.3)

        # Visiting team analysis
        visiting_data = plot_data.dropna(subset=[visiting_col, result])
        if not visiting_data.empty:
            ax2.scatter(visiting_data[visiting_col], visiting_data[result], alpha=0.6, color='red')
            self._add_regression_line(ax2, visiting_data[visiting_col], visiting_data[result])

        ax2.set_xlabel(self._format_name(visiting_col))
        ax2.set_ylabel(self._format_name(result))
        ax2.set_title('Visiting Team')
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f'{self._format_name(result)} vs {stat_pattern}', fontsize=14)
        plt.tight_layout()
        return fig

    def create_team_comparison(self,
                             teams: List[str],
                             features: List[str],
                             result: str,
                             figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Compare specific teams across multiple features.

        Args:
            teams (List[str]): List of team names
            features (List[str]): List of features to compare
            result (str): Result variable
            figsize (tuple): Figure size

        Returns:
            matplotlib.Figure: The created figure
        """
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

        colors = plt.cm.tab10(np.linspace(0, 1, len(teams)))

        for i, feature in enumerate(features):
            if i < len(axes):
                ax = axes[i]

                for j, team in enumerate(teams):
                    # Filter data for this team (either home or visiting)
                    team_data = self.data[
                        (self.data['result_home_team'] == team) |
                        (self.data['result_visiting_team'] == team)
                    ].copy()

                    # Determine if team is home or visiting and get appropriate stats
                    team_values = []
                    result_values = []

                    for _, row in team_data.iterrows():
                        if row['result_home_team'] == team:
                            # Team is playing at home
                            home_feature = feature.replace('visiting_', 'home_')
                            if home_feature in row:
                                team_values.append(row[home_feature])
                                result_values.append(row[result])
                        elif row['result_visiting_team'] == team:
                            # Team is visiting
                            visiting_feature = feature.replace('home_', 'visiting_')
                            if visiting_feature in row:
                                team_values.append(row[visiting_feature])
                                result_values.append(row[result])

                    if team_values and result_values:
                        ax.scatter(team_values, result_values,
                                 alpha=0.7, color=colors[j], label=team)

                ax.set_xlabel(self._format_name(feature))
                ax.set_ylabel(self._format_name(result))
                ax.set_title(self._format_name(feature))
                ax.grid(True, alpha=0.3)
                ax.legend()

        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f'Team Comparison: {self._format_name(result)}', fontsize=16)
        plt.tight_layout()
        return fig

    def find_strongest_correlations(self,
                                  result: str,
                                  top_n: int = 10) -> pd.DataFrame:
        """
        Find the features with strongest correlations to a result variable.

        Args:
            result (str): Result variable
            top_n (int): Number of top correlations to return

        Returns:
            pd.DataFrame: DataFrame with features and their correlations
        """
        if self.data is None:
            raise ValueError("No data loaded")

        features = self.analyzer.get_feature_columns()
        correlations = []

        plot_data = self._apply_filters()

        for feature in features:
            clean_data = plot_data.dropna(subset=[feature, result])
            if len(clean_data) > 1:
                corr = clean_data[feature].corr(clean_data[result])
                correlations.append({
                    'feature': feature,
                    'correlation': corr,
                    'abs_correlation': abs(corr),
                    'n_observations': len(clean_data)
                })

        # Convert to DataFrame and sort by absolute correlation
        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values('abs_correlation', ascending=False)

        return corr_df.head(top_n)

    def create_weekly_trend_plot(self,
                               feature: str,
                               result: str,
                               team: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Create a plot showing trends over weeks.

        Args:
            feature (str): Feature to analyze
            result (str): Result variable
            team (str, optional): Specific team to analyze
            figsize (tuple): Figure size

        Returns:
            matplotlib.Figure: The created figure
        """
        if 'week' not in self.data.columns:
            raise ValueError("Week column not found in data")

        plot_data = self._apply_filters()

        if team:
            plot_data = plot_data[
                (plot_data['result_home_team'] == team) |
                (plot_data['result_visiting_team'] == team)
            ]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Group by week and calculate means
        weekly_stats = plot_data.groupby('week').agg({
            feature: 'mean',
            result: 'mean'
        }).reset_index()

        # Plot trends
        ax1.plot(weekly_stats['week'], weekly_stats[feature], 'o-', color='blue')
        ax1.set_ylabel(self._format_name(feature))
        ax1.set_title(f'Weekly Trends{"" if not team else f" - {team}"}')
        ax1.grid(True, alpha=0.3)

        ax2.plot(weekly_stats['week'], weekly_stats[result], 'o-', color='red')
        ax2.set_xlabel('Week')
        ax2.set_ylabel(self._format_name(result))
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _apply_filters(self) -> pd.DataFrame:
        """Apply current filters to the data."""
        if self.data is None:
            raise ValueError("No data loaded")

        filtered_data = self.data.copy()
        for col, val in self.current_filters.items():
            if col in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[col] == val]

        return filtered_data

    def _add_regression_line(self, ax, x_data, y_data):
        """Add regression line and correlation to a plot."""
        if len(x_data) > 1:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            ax.plot(x_data, p(x_data), "r--", alpha=0.8)

            correlation = np.corrcoef(x_data, y_data)[0, 1]
            ax.text(0.05, 0.95, f'r = {correlation:.3f}\nn = {len(x_data)}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def _format_name(self, name: str) -> str:
        """Format column names for display."""
        formatted = name.replace('visiting_', '').replace('home_', '').replace('result_', '')
        formatted = formatted.replace('_', ' ').title()
        return formatted

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary information about the current dataset."""
        if self.data is None:
            return {"error": "No data loaded"}

        filtered_data = self._apply_filters()

        summary = {
            "total_games": len(self.data),
            "filtered_games": len(filtered_data),
            "years": sorted(self.data['year'].unique()) if 'year' in self.data.columns else [],
            "weeks": sorted(self.data['week'].unique()) if 'week' in self.data.columns else [],
            "teams": sorted(set(list(self.data['result_home_team'].unique()) +
                              list(self.data['result_visiting_team'].unique()))),
            "num_features": len(self.analyzer.get_feature_columns()),
            "num_results": len(self.analyzer.get_result_columns()),
            "current_filters": self.current_filters
        }

        return summary


if __name__ == "__main__":
    # Example usage
    print("NFL Interactive Data Explorer")
    print("="*40)

    # Create explorer and load data
    explorer = InteractiveExplorer()

    try:
        explorer.load_data(2024, 1)
        print("✓ Data loaded successfully")

        # Show data summary
        summary = explorer.get_data_summary()
        print(f"✓ Dataset: {summary['total_games']} games")
        print(f"✓ Features: {summary['num_features']}")
        print(f"✓ Results: {summary['num_results']}")

        # Find strongest correlations
        if summary['num_features'] > 0 and summary['num_results'] > 0:
            results = explorer.analyzer.get_result_columns()
            if results:
                correlations = explorer.find_strongest_correlations(results[0], top_n=5)
                print(f"\nTop correlations with {results[0]}:")
                for _, row in correlations.iterrows():
                    print(f"  {row['feature']}: {row['correlation']:.3f}")

    except Exception as e:
        print(f"Error: {e}")