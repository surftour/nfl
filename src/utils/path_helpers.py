"""
Path utility functions for constructing and parsing NFL data file paths.
"""
import re
from pathlib import Path
from typing import Optional, Tuple


def construct_data_path(year: int, week: int, file_type: str) -> str:
    """
    Construct a file path for NFL data files.
    
    Args:
        year (int): Four digit year (e.g., 2024)
        week (int): Week number (e.g., 1, 15)
        file_type (str): File type - one of "off", "def", or "results"
    
    Returns:
        str: File path in format "data/{year}/week{week:02d}/{file_type}.{ext}"
    
    Examples:
        >>> construct_data_path(2024, 1, "off")
        'data/2024/week01/off.csv'
        >>> construct_data_path(2024, 15, "results")
        'data/2024/week15/results.tsv'
    """
    # Determine file extension based on type
    extension = "tsv" if file_type == "results" else "csv"
    
    return f"data/{year:04d}/week{week:02d}/{file_type}.{extension}"


def extract_year_from_path(file_path: str) -> Optional[int]:
    """
    Extract year from a file path.
    
    Args:
        file_path (str): File path string
        
    Returns:
        Optional[int]: Four digit year if found, None otherwise
        
    Examples:
        >>> extract_year_from_path("data/2024/week01/off.csv")
        2024
        >>> extract_year_from_path("invalid/path")
        None
    """
    path_obj = Path(file_path)
    path_parts = path_obj.parts
    
    # Look for pattern: data/YYYY/weekXX
    for i, part in enumerate(path_parts):
        if part == 'data' and i + 1 < len(path_parts):
            year_part = path_parts[i + 1]
            
            # Check if year_part is a 4-digit number
            if re.match(r'^\d{4}$', year_part):
                return int(year_part)
    
    return None


def extract_week_from_path(file_path: str) -> Optional[int]:
    """
    Extract week number from a file path.
    
    Args:
        file_path (str): File path string
        
    Returns:
        Optional[int]: Week number if found, None otherwise
        
    Examples:
        >>> extract_week_from_path("data/2024/week01/off.csv")
        1
        >>> extract_week_from_path("data/2024/week15/def.csv")
        15
        >>> extract_week_from_path("invalid/path")
        None
    """
    path_obj = Path(file_path)
    path_parts = path_obj.parts
    
    # Look for pattern: data/YYYY/weekXX
    for i, part in enumerate(path_parts):
        if part == 'data' and i + 2 < len(path_parts):
            week_part = path_parts[i + 2]
            
            # Check if week_part matches weekNN pattern
            week_match = re.match(r'^week(\d+)$', week_part)
            if week_match:
                return int(week_match.group(1))
    
    return None


def extract_file_type_from_path(file_path: str) -> Optional[str]:
    """
    Extract file type from a file path.
    
    Args:
        file_path (str): File path string
        
    Returns:
        Optional[str]: File type ("off", "def", or "results") if found, None otherwise
        
    Examples:
        >>> extract_file_type_from_path("data/2024/week01/off.csv")
        'off'
        >>> extract_file_type_from_path("data/2024/week15/results.tsv")
        'results'
        >>> extract_file_type_from_path("invalid/path")
        None
    """
    path_obj = Path(file_path)
    filename = path_obj.stem  # Get filename without extension
    
    # Check if it's one of the expected file types
    if filename in ['off', 'def', 'results']:
        return filename
    
    return None


def extract_year_week_type_from_path(file_path: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """
    Extract year, week, and file type from a file path in one call.
    
    Args:
        file_path (str): File path string
        
    Returns:
        Tuple[Optional[int], Optional[int], Optional[str]]: (year, week, file_type) if found, (None, None, None) otherwise
        
    Examples:
        >>> extract_year_week_type_from_path("data/2024/week01/off.csv")
        (2024, 1, 'off')
        >>> extract_year_week_type_from_path("data/2024/week15/results.tsv")
        (2024, 15, 'results')
    """
    year = extract_year_from_path(file_path)
    week = extract_week_from_path(file_path)
    file_type = extract_file_type_from_path(file_path)
    
    return year, week, file_type