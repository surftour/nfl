import pandas as pd
import re
from pathlib import Path
from ..utils.path_helpers import construct_data_path, extract_year_week_type_from_path


def read_results_path(file_path):
    """
    Read weekly game results from a TSV file and convert to DataFrame
    
    Args:
        file_path: Path to the TSV file containing game results
        
    Returns:
        DataFrame with columns: date, result_visiting_team, result_visiting_score, result_home_team, result_home_score, result_status
    """
    games = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if line contains a date (format: Dec 10, 2023)
        if re.match(r'^[A-Za-z]{3} \d{1,2}, \d{4}$', line):
            date = line
            i += 1
            
            # Next line should be the visiting team and score
            if i < len(lines):
                visiting_line = lines[i].strip().split('\t')
                if len(visiting_line) >= 3:
                    visiting_team = visiting_line[0]
                    visiting_score = int(visiting_line[1])
                    status = visiting_line[2] if len(visiting_line) > 2 else ""
                    i += 1
                    
                    # Next line should be the home team and score
                    if i < len(lines):
                        home_line = lines[i].strip().split('\t')
                        if len(home_line) >= 2:
                            home_team = home_line[0]
                            home_score = int(home_line[1])
                            
                            games.append({
                                'date': date,
                                'result_visiting_team': visiting_team,
                                'result_visiting_score': visiting_score,
                                'result_home_team': home_team,
                                'result_home_score': home_score,
                                'result_status': status
                            })
        
        i += 1
    
    return pd.DataFrame(games)


def read_results(year: int, week: int):
    """
    Read weekly game results for a specific year and week.
    
    Args:
        year (int): Four digit year (e.g., 2024)
        week (int): Week number (e.g., 1, 15)
        
    Returns:
        DataFrame with columns: date, result_visiting_team, result_visiting_score, result_home_team, result_home_score, result_status, year, week
    """
    # Construct the file path using helper function
    file_path = construct_data_path(year, week, "results")
    
    # Convert to absolute path relative to project root
    base_path = Path(__file__).parent.parent.parent  # Go up to project root
    full_path = base_path / file_path
    
    if not full_path.exists():
        print(f"File not found: {full_path}")
        return None
    
    # Call the existing function with the constructed path
    df = read_results_path(str(full_path))
    
    # Ensure year and week columns are set (in case path parsing failed)
    if df is not None:
        df['year'] = year
        df['week'] = week
    
    return df