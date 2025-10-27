#!/usr/bin/env python3
"""
Weekly NFL Data Fetcher

This script is designed to be run on a weekly basis to:
1. Fetch all 2025 NFL games
2. Extract the current week number
3. Save current week's games to data/2025/week{N}/espn.games.csv
4. Save team stats to data/2025/week{N}/espn.stats.csv
5. Save Pro Football Reference season schedule to data/2025/week{N}/pfr.schedule.csv
6. Fetch metadata for games within next 7 days to data/2025/week{N}/pfr.metadata.csv
7. Compare to previous week and fetch details for newly completed games:
   - Save metadata to data/2025/week{N-1}/pfr.metadata.final.csv
   - Save statistics to data/2025/week{N-1}/pfr.statistics.csv
8. Save previous week's results to data/2025/week{N-1}/espn.results.csv
"""

import os
from datetime import datetime, timedelta
import pandas as pd
from src.api.espn import fetch_game_results
from src.api.pro_football_reference import fetch_season_schedule, fetch_games_details_batch
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


def get_games_within_days(schedule_df, days=7):
    """
    Filter schedule for games within the next N days

    Args:
        schedule_df: DataFrame with 'game_date' column
        days: Number of days to look ahead (default: 7)

    Returns:
        DataFrame with games within the specified timeframe
    """
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame()

    # Convert game_date to datetime
    schedule_df['game_date_dt'] = pd.to_datetime(schedule_df['game_date'], errors='coerce')

    # Get current date and date N days from now
    today = datetime.now()
    future_date = today + timedelta(days=days)

    # Filter for games within the timeframe
    upcoming = schedule_df[
        (schedule_df['game_date_dt'] >= today) &
        (schedule_df['game_date_dt'] <= future_date)
    ]

    return upcoming


def find_newly_completed_games(current_schedule_df, previous_schedule_df):
    """
    Find games that are in current schedule but weren't in previous schedule
    (indicating they were recently completed)

    Args:
        current_schedule_df: Current week's schedule DataFrame
        previous_schedule_df: Previous week's schedule DataFrame

    Returns:
        DataFrame with newly completed games
    """
    if previous_schedule_df is None or previous_schedule_df.empty:
        # No previous data to compare
        return pd.DataFrame()

    if current_schedule_df is None or current_schedule_df.empty:
        return pd.DataFrame()

    # Use boxscore_stats_link as unique identifier
    previous_links = set(previous_schedule_df['boxscore_stats_link'].dropna())
    current_links = set(current_schedule_df['boxscore_stats_link'].dropna())

    # Find games in current but not in previous (newly completed)
    new_links = current_links - previous_links

    # Return the newly completed games
    newly_completed = current_schedule_df[
        current_schedule_df['boxscore_stats_link'].isin(new_links)
    ]

    return newly_completed


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

    # Fetch and save Pro Football Reference season schedule
    pfr_schedule_path = f"/Users/tj/nfl/data/2025/week{current_week}/pfr.schedule.csv"
    print(f"Fetching Pro Football Reference season schedule...")
    pfr_schedule = save_to_csv(fetch_season_schedule, pfr_schedule_path, 2025)

    # Process PFR data: fetch metadata for upcoming games
    if pfr_schedule is not None and not pfr_schedule.empty:
        print("\nProcessing Pro Football Reference data...")

        # Step 1: Fetch metadata for games within next 7 days
        print("Looking for games within next 7 days...")
        upcoming_games = get_games_within_days(pfr_schedule, days=7)

        if not upcoming_games.empty:
            print(f"Found {len(upcoming_games)} games within next 7 days")
            upcoming_urls = upcoming_games['boxscore_stats_link'].dropna().tolist()

            if upcoming_urls:
                print("Fetching metadata for upcoming games...")
                upcoming_details = fetch_games_details_batch(
                    upcoming_urls,
                    include_metadata=True,
                    include_statistics=False
                )

                # Save upcoming games metadata
                if not upcoming_details['metadata'].empty:
                    pfr_metadata_path = f"/Users/tj/nfl/data/2025/week{current_week}/pfr.metadata.csv"
                    upcoming_details['metadata'].to_csv(pfr_metadata_path, index=False)
                    print(f"Saved upcoming games metadata to {pfr_metadata_path}")
        else:
            print("No games found within next 7 days")

        # Step 2: Compare to previous week and fetch details for newly completed games
        if current_week > 1:
            previous_week = current_week - 1
            previous_pfr_schedule_path = f"/Users/tj/nfl/data/2025/week{previous_week}/pfr.schedule.csv"

            if os.path.exists(previous_pfr_schedule_path):
                print(f"\nComparing to previous week ({previous_week}) schedule...")
                previous_pfr_schedule = pd.read_csv(previous_pfr_schedule_path)

                newly_completed = find_newly_completed_games(pfr_schedule, previous_pfr_schedule)

                if not newly_completed.empty:
                    print(f"Found {len(newly_completed)} newly completed games")
                    completed_urls = newly_completed['boxscore_stats_link'].dropna().tolist()

                    if completed_urls:
                        print("Fetching metadata and statistics for newly completed games...")
                        completed_details = fetch_games_details_batch(
                            completed_urls,
                            include_metadata=True,
                            include_statistics=True
                        )

                        # Save newly completed games metadata and statistics to previous week directory
                        previous_week_dir = f"/Users/tj/nfl/data/2025/week{previous_week}"

                        if not completed_details['metadata'].empty:
                            pfr_metadata_final_path = f"{previous_week_dir}/pfr.metadata.final.csv"
                            completed_details['metadata'].to_csv(pfr_metadata_final_path, index=False)
                            print(f"Saved completed games metadata to {pfr_metadata_final_path}")

                        if not completed_details['statistics'].empty:
                            pfr_statistics_path = f"{previous_week_dir}/pfr.statistics.csv"
                            completed_details['statistics'].to_csv(pfr_statistics_path, index=False)
                            print(f"Saved completed games statistics to {pfr_statistics_path}")
                else:
                    print("No newly completed games found")
            else:
                print(f"Previous week schedule not found at {previous_pfr_schedule_path}")
    else:
        print("Failed to fetch PFR schedule, skipping metadata processing")

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
