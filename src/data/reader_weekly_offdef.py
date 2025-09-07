import re
from pathlib import Path
from typing import Optional, List, Tuple

import pandas as pd
from ..utils.path_helpers import construct_data_path, extract_year_week_type_from_path


def _read_column_headers(file_path: Path):
    """Helper function to read and process column headers from first two lines"""
    with open(file_path, 'r', encoding='utf-8') as f:
        line1 = f.readline().strip()
        line2 = f.readline().strip()

    col1_parts = line1.split(',')
    col2_parts = line2.split(',')

    # Ensure both lines have the same number of columns
    if len(col1_parts) != len(col2_parts):
        print(f"Warning: First two lines have different number of columns in {file_path}")
        min_len = min(len(col1_parts), len(col2_parts))
        col1_parts = col1_parts[:min_len]
        col2_parts = col2_parts[:min_len]

    # Concatenate corresponding parts from both lines, clean up strings
    column_names = []
    for col1, col2 in zip(col1_parts, col2_parts):
        col1_clean = col1.strip().replace(' ', '').replace('&', 'Plus').replace('/', 'Per')
        col2_clean = col2.strip().replace(' ', '').replace('&', 'Plus').replace('/', 'Per')
        column_names.append(f"{col1_clean}{col2_clean}")

    return column_names

def _convert_data_types(df):
    """Helper function to convert DataFrame column data types"""
    for col in df.columns:
        if col in ['week', 'year']:
            continue

        has_decimals = df[col].astype(str).str.contains("\\.", na=False).any()
        is_text_only = df[col].astype(str).str.match(r'^[A-Za-z\s49]*$', na=False).all()

        try:
            if is_text_only:
                # explicitly cast as string
                df[col] = df[col].astype(str)
            elif has_decimals:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        except (ValueError, TypeError) as e:
            print(f"Error converting column '{col}': {e}")

    return df

def _process_csv_file(file_path: Path) -> Optional[pd.DataFrame]:
    """
    Helper function to read and process a CSV file with concatenated column names.

    Parameters:
        file_path (Path): Path to the CSV file

    Returns:
        pandas.DataFrame: Processed DataFrame, None if error
    """
    try:
        print(f"Processing CSV file: {file_path}")

        # Read and process column headers
        column_names = _read_column_headers(file_path)

        # Read the CSV starting from the third row as strings first
        df = pd.read_csv(file_path, skiprows=2, header=None, names=column_names, dtype=str)

        # Check if 'Tm' column exists and split out league totals
        if 'Tm' in df.columns:
            league_mask = df['Tm'].isin(['Avg Team', 'League Total', 'Avg Tm/G'])
            if league_mask.any():
                df = df[~league_mask].copy()
                # save because we might add some logic here at some point
                # leaguetotals = df[league_mask].copy()

        # Convert columns to appropriate data types
        df = _convert_data_types(df)

        # Extract year and week from file path and add columns if they don't exist
        year, week, _ = extract_year_week_type_from_path(str(file_path))
        if year is not None and 'year' not in df.columns:
            df['year'] = year
        if week is not None and 'week' not in df.columns:
            df['week'] = week

        return df

    except (IOError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def read_weekly_file(year: int, week: int, file_type: str) -> Optional[pd.DataFrame]:
    """
    Read a weekly CSV file from the data directory structure.

    Parameters:
        year (int): The year (e.g., 2020)
        week (int): The week number (e.g., 15)
        file_type (str): The file type/name (e.g., 'def', 'off', 'kicking')

    Returns:
        pandas.DataFrame: The CSV data if successful, None if file not found or error
    """
    # Use helper function to construct path
    file_path_str = construct_data_path(year, week, file_type)
    
    # Convert to absolute path relative to project root
    base_path = Path(__file__).parent.parent.parent  # Go up to project root
    file_path = base_path / file_path_str

    if file_path.exists():
        return _process_csv_file(file_path)

    print(f"File not found: {file_path}")
    return None


def read_all_weeks(year: int, file_type: str) -> Optional[pd.DataFrame]:
    """
    Read all weekly files of a specific type for a given year and combine them.

    Parameters:
        year (int): The year (e.g., 2020)
        file_type (str): The file type/name (e.g., 'def', 'off', 'kicking')

    Returns:
        pandas.DataFrame: Combined data from all weeks, None if no files found
    """
    base_path = Path(__file__).parent.parent.parent  # Go up to project root
    year_path = base_path / "data" / str(year)

    if not year_path.exists():
        print(f"Year directory not found: {year_path}")
        return None

    all_data = []

    # Look for week directories
    week_dirs = [d for d in year_path.iterdir() if d.is_dir() and d.name.startswith('week')]
    week_dirs.sort(key=lambda x: int(x.name.replace('week', '')))

    for week_dir in week_dirs:
        week_num = int(week_dir.name.replace('week', ''))
        file_path = week_dir / f"{file_type}.csv"

        if file_path.exists():
            df = _process_csv_file(file_path)
            if df is not None:
                all_data.append(df)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df

    print(f"No {file_type}.csv files found for year {year}")
    return None
