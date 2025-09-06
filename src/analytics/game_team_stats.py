import pandas as pd
from ..data.weekly_results_reader import read_results
from ..data.weekly_reader import read_weekly_file


def combine_game_results_with_team_stats(year, week, results_file_path=None):
    """
    Combine weekly game results with team offensive and defensive statistics.
    
    Args:
        year (int): Season year (e.g., 2023)
        week (int): Week number (e.g., 14)
        results_file_path (str): Path to results.tsv file. If None, uses year/week to construct path.
        
    Returns:
        DataFrame with each row as a game containing:
        - Game info: date, visiting_team, visiting_score, home_team, home_score, status, year, week
        - Visiting team stats: all off/def columns prefixed with 'visiting_'
        - Home team stats: all off/def columns prefixed with 'home_'
    """
    # Read game results using the new function
    if results_file_path is not None:
        games_df = read_results(results_file_path)
    else:
        games_df = read_results(year, week)
    if games_df is None or games_df.empty:
        if results_file_path:
            print(f"No game results found at {results_file_path}")
        else:
            print(f"No game results found for {year} week {week}")
        return None
    
    # Read offensive and defensive stats
    off_stats = read_weekly_file(year, week, 'off')
    def_stats = read_weekly_file(year, week, 'def')
    
    if off_stats is None or off_stats.empty:
        print(f"No offensive stats found for {year} week {week}")
        return None
        
    if def_stats is None or def_stats.empty:
        print(f"No defensive stats found for {year} week {week}")
        return None
    
    # Merge offensive and defensive stats on team name
    team_stats = pd.merge(
        off_stats, 
        def_stats, 
        on='Tm', 
        suffixes=('_off', '_def'),
        how='outer'
    )
    
    # Remove duplicate columns that exist in both datasets
    duplicate_cols = ['week_def', 'year_def'] if 'week_def' in team_stats.columns else []
    if duplicate_cols:
        team_stats = team_stats.drop(columns=duplicate_cols)
    
    # Get stat columns (excluding Tm and meta columns)
    stat_columns = [col for col in team_stats.columns 
                   if col not in ['Tm', 'week', 'year']]
    
    # Create copies with renamed columns for merging
    visiting_stats = team_stats[['Tm'] + stat_columns].copy()
    visiting_rename_dict = {col: f'visiting_{col}' for col in stat_columns}
    visiting_stats = visiting_stats.rename(columns=visiting_rename_dict)
    
    home_stats = team_stats[['Tm'] + stat_columns].copy()
    home_rename_dict = {col: f'home_{col}' for col in stat_columns}
    home_stats = home_stats.rename(columns=home_rename_dict)
    
    # Start with a clean copy of games
    result_df = games_df.copy()
    
    # Merge with visiting team stats
    result_df = pd.merge(
        result_df,
        visiting_stats,
        left_on='result_visiting_team',
        right_on='Tm',
        how='left'
    )
    result_df = result_df.drop(columns=['Tm'])
    
    # Merge with home team stats
    result_df = pd.merge(
        result_df,
        home_stats,
        left_on='result_home_team',
        right_on='Tm',
        how='left'
    )
    result_df = result_df.drop(columns=['Tm'])
    
    # Reorder columns for better readability
    base_cols = ['date', 'result_visiting_team', 'result_visiting_score', 'result_home_team', 'result_home_score', 'result_status']
    meta_cols = ['year', 'week'] if 'year' in result_df.columns else []
    visiting_cols = [col for col in result_df.columns if col.startswith('visiting_')]
    home_cols = [col for col in result_df.columns if col.startswith('home_')]
    
    ordered_cols = base_cols + meta_cols + visiting_cols + home_cols
    result_df = result_df[ordered_cols]
    
    return result_df


def save_combined_game_stats(year, week, filename=None, results_file_path=None):
    """
    Create combined game results with team stats and save to CSV.
    
    Args:
        year (int): Season year
        week (int): Week number
        filename (str): Output CSV filename. If None, auto-generates.
        results_file_path (str): Path to results.tsv file. If None, uses default.
        
    Returns:
        DataFrame if successful, None if failed
    """
    if filename is None:
        filename = f'data/{year}_week{week}_game_team_stats.csv'
    
    combined_df = combine_game_results_with_team_stats(year, week, results_file_path)
    
    if combined_df is not None and not combined_df.empty:
        combined_df.to_csv(filename, index=False)
        print(f"Combined game and team stats saved to '{filename}' ({len(combined_df)} games)")
        return combined_df
    
    return None