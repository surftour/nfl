from pathlib import Path
from typing import List


def list_available_weeks(year: int) -> List[int]:
    """
    List all available weeks for a given year.

    Parameters:
        year (int): The year (e.g., 2020)

    Returns:
        List[int]: List of available week numbers, empty list if year not found
    """
    base_path = Path(__file__).parent.parent.parent  # Go up to project root
    year_path = base_path / "data" / str(year)

    if not year_path.exists():
        print(f"Year directory not found: {year_path}")
        return []

    weeks = []

    # Look for week directories
    for item in year_path.iterdir():
        if item.is_dir() and item.name.startswith('week'):
            try:
                week_num = int(item.name.replace('week', ''))
                weeks.append(week_num)
            except ValueError:
                continue

    weeks.sort()
    return weeks


def list_available_file_types(year: int, week: int) -> List[str]:
    """
    List all available file types for a specific year and week.

    Parameters:
        year (int): The year (e.g., 2020)
        week (int): The week number (e.g., 15)

    Returns:
        List[str]: List of available file types (without .csv extension)
    """
    base_path = Path(__file__).parent.parent.parent  # Go up to project root
    week_path = base_path / "data" / str(year) / f"week{week}"

    if not week_path.exists():
        print(f"Week directory not found: {week_path}")
        return []

    file_types = []

    for file_path in week_path.glob("*.csv"):
        file_types.append(file_path.stem)  # Get filename without extension

    file_types.sort()
    return file_types