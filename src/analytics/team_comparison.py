import pandas as pd
from ..api.espn import fetch_teams, fetch_team_stats
from ..utils.data_io import save_to_csv

def create_team_comparison(year, selected_metrics=None):
    """
    Fetch stats for all NFL teams and create a comparison DataFrame

    Args:
        year: Season year (e.g., 2024)
        selected_metrics: List of metric names to include. If None, includes all available metrics.

    Returns:
        DataFrame with one row per team and selected metrics as columns
    """

    teams_df = fetch_teams()
    if teams_df is None or teams_df.empty:
        print("Failed to fetch teams data")
        return None

    comparison_data = []

    for _, team in teams_df.iterrows():
        team_id = team['id']
        team_name = team['displayName']
        team_abbr = team['abbreviation']

        print(f"Fetching stats for {team_name} (ID: {team_id})...")

        stats_df = fetch_team_stats(year, team_id)

        if stats_df is None or stats_df.empty:
            print(f"No stats found for {team_name}")
            continue

        # Create row for this team
        team_row = {
            'team_id': team_id,
            'team_name': team_name,
            'abbreviation': team_abbr
        }

        # Extract metrics
        if selected_metrics is None:
            # Grab all available metrics
            for _, stat in stats_df.iterrows():
                metric_name = stat['name']
                team_row[metric_name] = stat['value']
        else:
            # Extract only selected metrics
            for metric in selected_metrics:
                matching_stats = stats_df[stats_df['name'] == metric]
                if not matching_stats.empty:
                    team_row[metric] = matching_stats.iloc[0]['value']
                else:
                    team_row[metric] = None

        comparison_data.append(team_row)

    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df

def save_team_comparison(year, filename=None, selected_metrics=None):
    """
    Create team comparison and save to CSV
    
    Args:
        year: Season year
        filename: Output CSV filename. If None, auto-generates based on year.
        selected_metrics: List of metrics to include
    
    Returns:
        DataFrame if successful
    """
    if filename is None:
        filename = f'data/nfl_team_comparison_{year}.csv'

    return save_to_csv(create_team_comparison, filename, year, selected_metrics)
