"""
Pro Football Reference API Module

This module provides functions to scrape NFL data from Pro Football Reference
using the nflscraPy library.

Functions:
    - fetch_season_schedule: Get all games for a season
    - fetch_game_metadata: Get metadata for a specific game (venue, weather, etc.)
    - fetch_game_statistics: Get statistics for a specific game
    - fetch_season_with_details: Get season schedule + detailed data for all games
"""

import pandas as pd

try:
    import nflscraPy
except ImportError:
    print("Error: nflscraPy library not installed. Run: pip install nflscraPy")
    nflscraPy = None


def fetch_season_schedule(season):
    """
    Fetch all games for a season from Pro Football Reference

    Args:
        season (int): Season year (e.g., 2024). Data available from 1970 onwards.

    Returns:
        pandas.DataFrame with columns including:
            - season: Season year
            - week: Week number
            - game_day_of_week: Day of week
            - game_date: Date of game
            - boxscore_word: Game type (e.g., 'Final', 'Playoffs')
            - winning_team: Winning team name
            - winning_team_points: Points scored by winner
            - losing_team: Losing team name
            - losing_team_points: Points scored by loser
            - boxscore_stats_link: URL to detailed boxscore (use for fetch_game_* functions)
            - home_team: Home team indicator
            - winning_team_abbr: Winning team abbreviation
            - losing_team_abbr: Losing team abbreviation

    Example:
        >>> schedule = fetch_season_schedule(2024)
        >>> print(schedule[['week', 'game_date', 'winning_team', 'losing_team']])
    """
    if nflscraPy is None:
        print("Error: nflscraPy not available")
        return pd.DataFrame()

    try:
        df = nflscraPy._gamelogs(season)
        return df
    except Exception as e:
        print(f"Error fetching season schedule for {season}: {e}")
        return pd.DataFrame()


def fetch_game_metadata(boxscore_url):
    """
    Fetch game metadata (venue, weather, betting lines, etc.) for a specific game

    Args:
        boxscore_url (str): Boxscore URL from fetch_season_schedule's 'boxscore_stats_link' column
            Format: 'https://www.pro-football-reference.com/boxscores/YYYYMMDDTEAM.htm'

    Returns:
        pandas.DataFrame with columns including:
            - boxscore_stats_link: Game identifier URL
            - vegas_line: Betting line
            - over_under: Over/under line
            - roof: Roof type (outdoors, dome, etc.)
            - surface: Playing surface
            - duration: Game duration
            - attendance: Attendance
            - weather: Weather conditions (temperature, wind, etc.)
            - start_time: Game start time

    Example:
        >>> schedule = fetch_season_schedule(2024)
        >>> boxscore_url = schedule.iloc[0]['boxscore_stats_link']
        >>> metadata = fetch_game_metadata(boxscore_url)
    """
    if nflscraPy is None:
        print("Error: nflscraPy not available")
        return pd.DataFrame()

    try:
        df = nflscraPy._gamelog_metadata(boxscore_url)
        return df
    except Exception as e:
        print(f"Error fetching game metadata for {boxscore_url}: {e}")
        return pd.DataFrame()


def fetch_game_statistics(boxscore_url):
    """
    Fetch game statistics (team stats, scoring, turnovers) for a specific game

    Args:
        boxscore_url (str): Boxscore URL from fetch_season_schedule's 'boxscore_stats_link' column
            Format: 'https://www.pro-football-reference.com/boxscores/YYYYMMDDTEAM.htm'

    Returns:
        pandas.DataFrame with columns including:
            - boxscore_stats_link: Game identifier URL
            - team: Team abbreviation
            - first_downs: First downs
            - rush_yards: Rushing yards
            - rush_att: Rushing attempts
            - pass_yards: Passing yards
            - pass_cmp: Passes completed
            - pass_att: Passes attempted
            - sacks: Sacks allowed
            - net_pass_yards: Net passing yards
            - total_yards: Total yards
            - fumbles: Fumbles
            - fumbles_lost: Fumbles lost
            - turnovers: Total turnovers
            - penalties: Penalties
            - penalty_yards: Penalty yards
            - third_down_conversions: Third down conversions
            - fourth_down_conversions: Fourth down conversions
            - time_of_possession: Time of possession

    Example:
        >>> schedule = fetch_season_schedule(2024)
        >>> boxscore_url = schedule.iloc[0]['boxscore_stats_link']
        >>> stats = fetch_game_statistics(boxscore_url)
    """
    if nflscraPy is None:
        print("Error: nflscraPy not available")
        return pd.DataFrame()

    try:
        df = nflscraPy._gamelog_statistics(boxscore_url)
        return df
    except Exception as e:
        print(f"Error fetching game statistics for {boxscore_url}: {e}")
        return pd.DataFrame()


def fetch_games_details_batch(boxscore_urls, include_metadata=True, include_statistics=True):
    """
    Fetch metadata and/or statistics for multiple games given their boxscore URLs

    This is a helper function that processes a list of boxscore URLs and fetches
    detailed data for each game.

    WARNING: This function makes HTTP requests with rate limiting (3.5-5.5 sec per game).
    Processing time = (number of URLs) * (4-5 seconds average).

    Args:
        boxscore_urls (list): List of boxscore URLs to process
        include_metadata (bool): Whether to fetch metadata for each game (default: True)
        include_statistics (bool): Whether to fetch statistics for each game (default: True)

    Returns:
        dict with keys:
            - 'metadata': DataFrame with metadata for all games (if include_metadata=True)
            - 'statistics': DataFrame with statistics for all games (if include_statistics=True)

    Example:
        >>> urls = ['https://www.pro-football-reference.com/boxscores/202212180jax.htm',
        ...         'https://www.pro-football-reference.com/boxscores/202212181buf.htm']
        >>> details = fetch_games_details_batch(urls, include_metadata=True, include_statistics=True)
        >>> print(details['metadata'].shape)
        >>> print(details['statistics'].shape)
    """
    if nflscraPy is None:
        print("Error: nflscraPy not available")
        return {
            'metadata': pd.DataFrame(),
            'statistics': pd.DataFrame()
        }

    metadata_list = []
    statistics_list = []
    total_games = len(boxscore_urls)

    # Fetch details for each game
    for idx, boxscore_url in enumerate(boxscore_urls, 1):
        if not boxscore_url:
            continue

        print(f"Processing game {idx}/{total_games}: {boxscore_url}")

        if include_metadata:
            meta = fetch_game_metadata(boxscore_url)
            if not meta.empty:
                metadata_list.append(meta)

        if include_statistics:
            stats = fetch_game_statistics(boxscore_url)
            if not stats.empty:
                statistics_list.append(stats)

    # Combine all metadata and statistics
    # Filter out empty DataFrames to avoid FutureWarning
    metadata_list = [df for df in metadata_list if not df.empty]
    statistics_list = [df for df in statistics_list if not df.empty]

    metadata_df = pd.concat(metadata_list, ignore_index=True) if metadata_list else pd.DataFrame()
    statistics_df = pd.concat(statistics_list, ignore_index=True) if statistics_list else pd.DataFrame()

    print(f"\nCompleted processing {total_games} games")
    print(f"Metadata: {len(metadata_df)} records")
    print(f"Statistics: {len(statistics_df)} records")

    return {
        'metadata': metadata_df,
        'statistics': statistics_df
    }


def fetch_season_with_details(season, include_metadata=True, include_statistics=True,
                               max_games=None):
    """
    Fetch season schedule and optionally fetch detailed data for each game

    This is a batch processing function that combines fetch_season_schedule with
    fetch_game_metadata and/or fetch_game_statistics for all games in a season.

    WARNING: This function makes many HTTP requests (2-3 per game). A full season
    (~280 games including playoffs) will take 30-45 minutes due to rate limiting.
    Use max_games parameter to test with a subset.

    Args:
        season (int): Season year (e.g., 2024)
        include_metadata (bool): Whether to fetch metadata for each game (default: True)
        include_statistics (bool): Whether to fetch statistics for each game (default: True)
        max_games (int, optional): Maximum number of games to process (for testing)

    Returns:
        dict with keys:
            - 'schedule': DataFrame from fetch_season_schedule
            - 'metadata': DataFrame with metadata for all games (if include_metadata=True)
            - 'statistics': DataFrame with statistics for all games (if include_statistics=True)

    Example:
        >>> # Fetch first 5 games for testing
        >>> data = fetch_season_with_details(2024, max_games=5)
        >>> print(data['schedule'].shape)
        >>> print(data['metadata'].shape)
        >>> print(data['statistics'].shape)

        >>> # Fetch full season (will take 30-45 minutes)
        >>> data = fetch_season_with_details(2024)
    """
    if nflscraPy is None:
        print("Error: nflscraPy not available")
        return {
            'schedule': pd.DataFrame(),
            'metadata': pd.DataFrame(),
            'statistics': pd.DataFrame()
        }

    # First, get all games for the season
    print(f"Fetching season schedule for {season}...")
    schedule = fetch_season_schedule(season)

    if schedule.empty:
        print(f"No schedule data found for season {season}")
        return {
            'schedule': schedule,
            'metadata': pd.DataFrame(),
            'statistics': pd.DataFrame()
        }

    # Limit to max_games if specified
    if max_games:
        schedule = schedule.head(max_games)
        print(f"Limited to {max_games} games for processing")

    # Extract boxscore URLs from schedule
    boxscore_urls = schedule['boxscore_stats_link'].dropna().tolist()

    # Fetch details for all games using helper function
    details = fetch_games_details_batch(
        boxscore_urls,
        include_metadata=include_metadata,
        include_statistics=include_statistics
    )

    print(f"Schedule: {len(schedule)} records")

    return {
        'schedule': schedule,
        'metadata': details['metadata'],
        'statistics': details['statistics']
    }


if __name__ == "__main__":
    """
    Example usage - demonstrates the key functions
    """
    from ..utils.data_io import save_to_csv

    print("Pro Football Reference API Examples")
    print("=" * 50)

    # Example 1: Fetch season schedule
    print("\n1. Fetching 2024 season schedule...")
    schedule_df = save_to_csv(fetch_season_schedule, 'data/pfr_2024_schedule.csv', 2024)

    if schedule_df is not None and not schedule_df.empty:
        print(f"   Retrieved {len(schedule_df)} games")
        print(f"   Sample data:\n{schedule_df[['week', 'game_date', 'winning_team', 'losing_team']].head()}")

        # Example 2: Fetch metadata for first game
        print("\n2. Fetching metadata for first game...")
        first_game_url = schedule_df.iloc[0]['boxscore_stats_link']
        metadata_df = save_to_csv(fetch_game_metadata, 'data/pfr_game_metadata_sample.csv',
                                  first_game_url)

        if metadata_df is not None:
            print(f"   Retrieved metadata columns: {list(metadata_df.columns)}")

        # Example 3: Fetch statistics for first game
        print("\n3. Fetching statistics for first game...")
        stats_df = save_to_csv(fetch_game_statistics, 'data/pfr_game_stats_sample.csv',
                               first_game_url)

        if stats_df is not None:
            print(f"   Retrieved stats for {len(stats_df)} teams")

        # Example 4: Fetch first 3 games with full details (testing batch processing)
        print("\n4. Fetching first 3 games with full details...")
        print("   (This will take ~30-45 seconds due to rate limiting)")
        detailed_data = fetch_season_with_details(2024, max_games=3)

        # Save each component
        if not detailed_data['schedule'].empty:
            detailed_data['schedule'].to_csv('data/pfr_detailed_schedule_sample.csv', index=False)
        if not detailed_data['metadata'].empty:
            detailed_data['metadata'].to_csv('data/pfr_detailed_metadata_sample.csv', index=False)
        if not detailed_data['statistics'].empty:
            detailed_data['statistics'].to_csv('data/pfr_detailed_stats_sample.csv', index=False)

        print("\n   All sample data saved to data/ directory")
    else:
        print("   Failed to retrieve schedule data")

    print("\n" + "=" * 50)
    print("Examples complete!")
