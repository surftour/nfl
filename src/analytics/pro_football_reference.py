"""
Pro Football Reference Analytics Module

This module provides higher-level analytics functions for Pro Football Reference data.

Functions:
    - fetch_week_details: Fetch metadata and statistics for all games in a specific week
    - save_week_details_to_csv: Fetch and save week details to CSV files
    - get_cumulative_team_stats: Get cumulative team statistics from week 1 to N
"""

import os
import glob
import pandas as pd
from src.api.pro_football_reference import fetch_games_details_batch
from src.utils.data_io import ensure_directory_exists, version_existing_file


def fetch_week_details(week, season=2025, include_metadata=True, include_statistics=True, base_dir="/Users/tj/nfl/data"):
    """
    Fetch metadata and statistics for all games in a specific week

    This function:
    1. Finds the most recent pfr.schedule.csv file in data/{season}/week* directories
    2. Filters for the specified week
    3. Extracts all boxscore_stats_link URLs for that week
    4. Fetches metadata and/or statistics for all games in that week

    WARNING: This function makes HTTP requests with rate limiting (3.5-5.5 sec per game).
    Processing time depends on number of games in the week (typically 15-16 games = ~1-2 minutes).

    Args:
        week (int): Week number to fetch (e.g., 1, 2, 3, ...)
        season (int): Season year (default: 2025)
        include_metadata (bool): Whether to fetch metadata for each game (default: True)
        include_statistics (bool): Whether to fetch statistics for each game (default: True)
        base_dir (str): Base data directory (default: "/Users/tj/nfl/data")

    Returns:
        dict with keys:
            - 'schedule': DataFrame with games from the specified week
            - 'metadata': DataFrame with metadata for all games (if include_metadata=True)
            - 'statistics': DataFrame with statistics for all games (if include_statistics=True)
            - 'source_file': Path to the schedule file used

    Example:
        >>> # Fetch metadata and stats for week 5
        >>> details = fetch_week_details(5)
        >>> print(f"Found {len(details['schedule'])} games in week 5")
        >>> print(f"Metadata: {len(details['metadata'])} records")
        >>> print(f"Statistics: {len(details['statistics'])} records")

        >>> # Save to CSV
        >>> details['metadata'].to_csv('data/week5_metadata.csv', index=False)
        >>> details['statistics'].to_csv('data/week5_statistics.csv', index=False)
    """
    # Find all pfr.schedule.csv files (including versioned ones)
    season_dir = f"{base_dir}/{season}"
    schedule_files = glob.glob(f"{season_dir}/week*/pfr.schedule*.csv")

    if not schedule_files:
        print(f"No pfr.schedule.csv files found in {season_dir}/week*/")
        return {
            'schedule': pd.DataFrame(),
            'metadata': pd.DataFrame(),
            'statistics': pd.DataFrame(),
            'source_file': None
        }

    # Sort by modification time, most recent first
    schedule_files.sort(key=os.path.getmtime, reverse=True)
    most_recent_file = schedule_files[0]

    print(f"Using schedule file: {most_recent_file}")

    try:
        # Read the schedule
        schedule_df = pd.read_csv(most_recent_file)

        if schedule_df.empty:
            print("Schedule file is empty")
            return {
                'schedule': pd.DataFrame(),
                'metadata': pd.DataFrame(),
                'statistics': pd.DataFrame(),
                'source_file': most_recent_file
            }

        # Filter for the specified week
        if 'week' not in schedule_df.columns:
            print("'week' column not found in schedule file")
            return {
                'schedule': pd.DataFrame(),
                'metadata': pd.DataFrame(),
                'statistics': pd.DataFrame(),
                'source_file': most_recent_file
            }

        week_games = schedule_df[schedule_df['week'] == week].copy()

        if week_games.empty:
            print(f"No games found for week {week} in schedule")
            return {
                'schedule': week_games,
                'metadata': pd.DataFrame(),
                'statistics': pd.DataFrame(),
                'source_file': most_recent_file
            }

        print(f"Found {len(week_games)} games for week {week}")

        # Extract boxscore URLs
        if 'boxscore_stats_link' not in week_games.columns:
            print("'boxscore_stats_link' column not found in schedule file")
            return {
                'schedule': week_games,
                'metadata': pd.DataFrame(),
                'statistics': pd.DataFrame(),
                'source_file': most_recent_file
            }

        boxscore_urls = week_games['boxscore_stats_link'].dropna().tolist()

        if not boxscore_urls:
            print(f"No boxscore URLs found for week {week}")
            return {
                'schedule': week_games,
                'metadata': pd.DataFrame(),
                'statistics': pd.DataFrame(),
                'source_file': most_recent_file
            }

        print(f"Fetching details for {len(boxscore_urls)} games...")

        # Fetch details for all games in the week
        details = fetch_games_details_batch(
            boxscore_urls,
            include_metadata=include_metadata,
            include_statistics=include_statistics
        )

        return {
            'schedule': week_games,
            'metadata': details['metadata'],
            'statistics': details['statistics'],
            'source_file': most_recent_file
        }

    except Exception as e:
        print(f"Error processing schedule file: {e}")
        return {
            'schedule': pd.DataFrame(),
            'metadata': pd.DataFrame(),
            'statistics': pd.DataFrame(),
            'source_file': most_recent_file
        }


def save_week_details_to_csv(week, season=2025, include_metadata=True, include_statistics=True, base_dir="/Users/tj/nfl/data"):
    """
    Fetch metadata and statistics for all games in a specific week and save to CSV files

    This function calls fetch_week_details and saves the results to CSV files in the
    data/{season}/week{week:02d}/ directory with versioning.

    Files saved:
        - pfr.metadata.csv: Game metadata (venue, weather, betting lines)
        - pfr.statistics.csv: Game statistics (yards, turnovers, etc.)

    Args:
        week (int): Week number to fetch (e.g., 1, 2, 3, ...)
        season (int): Season year (default: 2025)
        include_metadata (bool): Whether to fetch and save metadata (default: True)
        include_statistics (bool): Whether to fetch and save statistics (default: True)
        base_dir (str): Base data directory (default: "/Users/tj/nfl/data")

    Returns:
        dict with keys:
            - 'metadata_path': Path to saved metadata file (None if not saved)
            - 'statistics_path': Path to saved statistics file (None if not saved)
            - 'details': The full details dict from fetch_week_details

    Example:
        >>> # Fetch and save metadata and stats for week 5
        >>> result = save_week_details_to_csv(5)
        >>> print(f"Metadata saved to: {result['metadata_path']}")
        >>> print(f"Statistics saved to: {result['statistics_path']}")
    """
    # Fetch the week details
    print(f"Fetching details for week {week}, season {season}...")
    details = fetch_week_details(
        week=week,
        season=season,
        include_metadata=include_metadata,
        include_statistics=include_statistics,
        base_dir=base_dir
    )

    # Create week directory path
    week_dir = f"{base_dir}/{season}/week{week:02d}"

    metadata_path = None
    statistics_path = None

    # Save metadata if available
    if include_metadata and not details['metadata'].empty:
        metadata_path = f"{week_dir}/pfr.metadata.csv"
        ensure_directory_exists(metadata_path)
        version_existing_file(metadata_path)
        details['metadata'].to_csv(metadata_path, index=False)
        print(f"Saved metadata to {metadata_path}")
    elif include_metadata:
        print("No metadata to save (empty DataFrame)")

    # Save statistics if available
    if include_statistics and not details['statistics'].empty:
        statistics_path = f"{week_dir}/pfr.statistics.csv"
        ensure_directory_exists(statistics_path)
        version_existing_file(statistics_path)
        details['statistics'].to_csv(statistics_path, index=False)
        print(f"Saved statistics to {statistics_path}")
    elif include_statistics:
        print("No statistics to save (empty DataFrame)")

    return {
        'metadata_path': metadata_path,
        'statistics_path': statistics_path,
        'details': details
    }


def get_cumulative_team_stats(end_week, season=2025, base_dir="/Users/tj/nfl/data"):
    """
    Combine weekly team statistics from week 1 to end_week to get cumulative per-team totals

    This function reads pfr.statistics.csv files from weeks 1 to end_week,
    and aggregates the statistics by team (using the 'alias' column).

    Args:
        end_week (int): Ending week number (e.g., 5 means weeks 1-5)
        season (int): Season year (default: 2025)
        base_dir (str): Base data directory (default: "/Users/tj/nfl/data")

    Returns:
        pandas.DataFrame with cumulative team statistics, including:
            - alias: Team abbreviation
            - market: Team market/city
            - name: Team name
            - games_played: Number of games in the period
            - All numeric statistics summed across games
            - Non-summable columns (like percentages) are averaged

    Example:
        >>> # Get cumulative stats for weeks 1-5
        >>> cumulative = get_cumulative_team_stats(5)
        >>> print(cumulative[['alias', 'games_played', 'rush_yds', 'pass_yds', 'total_yds']])

        >>> # Save to CSV
        >>> cumulative.to_csv('data/cumulative_weeks_1_5.csv', index=False)
    """
    import pandas as pd

    all_weeks_data = []
    start_week = 1

    # Read statistics files for each week
    for week in range(start_week, end_week + 1):
        week_file = f"{base_dir}/{season}/week{week:02d}/pfr.statistics.csv"

        if not os.path.exists(week_file):
            print(f"Warning: File not found: {week_file}")
            continue

        try:
            week_df = pd.read_csv(week_file)
            if not week_df.empty:
                week_df['week'] = week  # Add week column for tracking
                all_weeks_data.append(week_df)
                print(f"Loaded week {week}: {len(week_df)} team records")
        except Exception as e:
            print(f"Error reading {week_file}: {e}")
            continue

    if not all_weeks_data:
        print("No data found for specified weeks")
        return pd.DataFrame()

    # Combine all weeks
    combined_df = pd.concat(all_weeks_data, ignore_index=True)
    print(f"\nCombined weeks 1-{end_week}: {len(combined_df)} total team-game records")

    # Define which columns to sum vs average vs keep first
    # Columns to exclude from aggregation (identifiers, non-numeric, or links)
    exclude_cols = ['season', 'event_date', 'nano', 'boxscore_stats_link', 'week']

    # Columns that should be averaged (percentages and rates)
    average_cols = ['pass_cmp_pct', 'passer_rating', 'third_down_conv_pct', 'fourth_down_conv_pct']

    # Columns to keep (take first value - these don't change)
    keep_first_cols = ['market', 'name', 'alias']

    # All other numeric columns will be summed
    numeric_cols = combined_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    sum_cols = [col for col in numeric_cols if col not in average_cols and col not in exclude_cols and col != 'week']

    # Build aggregation dictionary
    agg_dict = {}

    # Keep first for identifier columns
    for col in keep_first_cols:
        if col in combined_df.columns:
            agg_dict[col] = 'first'

    # Sum for numeric columns
    for col in sum_cols:
        if col in combined_df.columns:
            agg_dict[col] = 'sum'

    # Average for percentage/rate columns
    for col in average_cols:
        if col in combined_df.columns:
            agg_dict[col] = 'mean'

    # Count games played
    agg_dict['week'] = 'count'

    # Group by team alias and aggregate
    cumulative_df = combined_df.groupby('alias', as_index=False).agg(agg_dict)

    # Rename the 'week' column to 'games_played'
    cumulative_df.rename(columns={'week': 'games_played'}, inplace=True)

    print(f"\nAggregated to {len(cumulative_df)} teams")
    print(f"Columns: {list(cumulative_df.columns)}")

    return cumulative_df


if __name__ == "__main__":
    """
    Command-line interface for fetching week details and cumulative stats
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch and save Pro Football Reference metadata and statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch and save data for week 5 of 2025 season
  python -m src.analytics.pro_football_reference --week 5

  # Fetch for a different season
  python -m src.analytics.pro_football_reference --week 5 --season 2024

  # Fetch only metadata (no statistics)
  python -m src.analytics.pro_football_reference --week 5 --no-statistics

  # Get cumulative team stats from weeks 1-5
  python -m src.analytics.pro_football_reference --cumulative 5

  # Get cumulative stats and save to specific file
  python -m src.analytics.pro_football_reference --cumulative 5 --output data/cumulative_week05.csv
        """
    )

    parser.add_argument(
        '--week',
        type=int,
        help='Week number to fetch (e.g., 1, 2, 3, ...)'
    )

    parser.add_argument(
        '--cumulative',
        type=int,
        metavar='END_WEEK',
        help='Get cumulative stats from week 1 to END_WEEK (e.g., --cumulative 5 for weeks 1-5)'
    )

    parser.add_argument(
        '--season',
        type=int,
        default=2025,
        help='Season year (default: 2025)'
    )

    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Skip fetching metadata (only applies to --week mode)'
    )

    parser.add_argument(
        '--no-statistics',
        action='store_true',
        help='Skip fetching statistics (only applies to --week mode)'
    )

    parser.add_argument(
        '--base-dir',
        type=str,
        default="/Users/tj/nfl/data",
        help='Base data directory (default: /Users/tj/nfl/data)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file for cumulative stats (only applies to --cumulative mode)'
    )

    args = parser.parse_args()

    # Check that either --week or --cumulative is provided
    if not args.week and args.cumulative is None:
        parser.error("Either --week or --cumulative must be specified")

    if args.week and args.cumulative is not None:
        parser.error("Cannot use both --week and --cumulative at the same time")

    # Handle weekly fetch mode
    if args.week:
        print("=" * 60)
        print(f"Fetching Pro Football Reference data for Week {args.week}, Season {args.season}")
        print("=" * 60)

        result = save_week_details_to_csv(
            week=args.week,
            season=args.season,
            include_metadata=not args.no_metadata,
            include_statistics=not args.no_statistics,
            base_dir=args.base_dir
        )

        print("\n" + "=" * 60)
        print("Complete!")
        print(f"  Metadata: {result['metadata_path'] or 'Not saved'}")
        print(f"  Statistics: {result['statistics_path'] or 'Not saved'}")
        print("=" * 60)

    # Handle cumulative stats mode
    elif args.cumulative is not None:
        end_week = args.cumulative
        print("=" * 60)
        print(f"Getting cumulative team stats for weeks 1-{end_week}, Season {args.season}")
        print("=" * 60)

        cumulative_df = get_cumulative_team_stats(
            end_week=end_week,
            season=args.season,
            base_dir=args.base_dir
        )

        if not cumulative_df.empty:
            # Determine output file
            if args.output:
                output_file = args.output
            else:
                output_file = f"{args.base_dir}/{args.season}/week{end_week:02d}/pfr.cumulative.csv"

            # Save to CSV
            ensure_directory_exists(output_file)
            cumulative_df.to_csv(output_file, index=False)
            print(f"\n{'=' * 60}")
            print(f"Saved cumulative stats to: {output_file}")
            print(f"{'=' * 60}")
        else:
            print("\nNo cumulative data to save")
