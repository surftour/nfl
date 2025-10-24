#!/usr/bin/env python3
"""
Weekly NFL Data Fetcher

This script is designed to be run on a weekly basis to:
1. Fetch all 2025 NFL games
2. Extract the current week number
3. Save current week's games to data/2025/week{N}/espn.games.csv
4. Save team stats to data/2025/week{N}/espn.stats.csv
5. Save previous week's results to data/2025/week{N-1}/espn.results.csv
"""

import os
from src.api.espn import fetch_game_results
from src.utils.data_io import save_to_csv
from src.analytics.team_comparison import save_team_comparison


def get_current_week(games_df):
    """
    Extract the current week number from the games dataframe.
    Returns the maximum week number found in the data.
    """
    if games_df is None or games_df.empty:
        raise ValueError("No games data available")

    current_week = int(games_df['week'].max())
    return current_week


def ensure_directory_exists(filepath):
    """Create directory for the filepath if it doesn't exist"""
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def main():
    """Main function to fetch and save weekly NFL data"""

    print("Fetching current week 2025 NFL games...")
    all_games = fetch_game_results(2025)

    if all_games is None or all_games.empty:
        print("ERROR: Failed to fetch games data")
        return

    # Extract current week
    current_week = get_current_week(all_games)
    print(f"Current week detected: {current_week}")

    # Save current week games
    current_week_path = f"/Users/tj/nfl/data/2025/week{current_week}/espn.games.csv"
    ensure_directory_exists(current_week_path)

    # Check if already run for this week
    if os.path.exists(current_week_path):
        print(f"Data for week {current_week} already exists at {current_week_path}")
        print("Script has already been run for this week. Exiting.")
        return

    print(f"Saving current week ({current_week}) games...")
    save_to_csv(fetch_game_results, current_week_path, 2025)

    # Save team stats for the current week
    stats_path = f"/Users/tj/nfl/data/2025/week{current_week}/espn.stats.csv"
    print(f"Saving team stats for week {current_week}...")
    save_team_comparison(2025, stats_path)

    # Save previous week results (if week > 1)
    if current_week > 1:
        previous_week = current_week - 1
        previous_week_path = f"/Users/tj/nfl/data/2025/week{previous_week}/espn.results.csv"
        ensure_directory_exists(previous_week_path)
        print(f"Saving previous week ({previous_week}) results...")
        save_to_csv(fetch_game_results, previous_week_path, 2025, previous_week)
    else:
        print("Current week is 1, no previous week to save")

    print("Weekly data fetch complete!")


if __name__ == "__main__":
    main()
