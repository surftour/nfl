# Weekly Data Reader Functions Documentation

## Overview

The `reader_weekly_offdef.py` module provides functions to read NFL weekly data from CSV files stored in the project's `data/` directory structure. Files are organized as `data/YYYY/weekXX/filename.csv`.

## Functions

### read_weekly_file(year, week, file_type, normalize=True)

Reads a specific weekly CSV file from the data directory structure with optional stat normalization.

**Parameters:**
- `year` (int): The year (e.g., 2020)
- `week` (int): The week number (e.g., 15)
- `file_type` (str): The file type/name (e.g., 'def', 'off', 'kicking')
- `normalize` (bool): Whether to normalize stats by games played (default: True)

**Returns:**
- `pandas.DataFrame`: The CSV data if successful, `None` if file not found or error

**Example:**
```python
from src.data.reader_weekly_offdef import read_weekly_file

# Read defensive stats for week 15 of 2020 season (normalized by default)
def_data = read_weekly_file(2020, 15, 'def')

# Read raw defensive stats without normalization
def_data_raw = read_weekly_file(2020, 15, 'def', normalize=False)
```

**Note:** When `normalize=True`, the following columns are automatically converted to per-game averages (with "PerGame" suffix) and replace the original columns: PF, Yds, TotYdsPlusTOPly, TotYdsPlusTOTO, FL, 1stD, PassingCmp, PassingAtt, PassingYds, PassingTD, PassingInt, Passing1stD, RushingAtt, RushingYds, RushingTD, Rushing1stD, PenaltiesPen, PenaltiesYds, Penalties1stPy.

### read_all_weeks(year, file_type)

Reads all weekly files of a specific type for a given year and combines them into a single DataFrame.

**Parameters:**
- `year` (int): The year (e.g., 2020)
- `file_type` (str): The file type/name (e.g., 'def', 'off', 'kicking')

**Returns:**
- `pandas.DataFrame`: Combined data from all weeks with added `week` and `year` columns, `None` if no files found

**Example:**
```python
from src.data.reader_weekly_offdef import read_all_weeks

# Read all defensive stats for 2020 season
all_def_data = read_all_weeks(2020, 'def')
```

### list_available_weeks(year)

Lists all available weeks for a given year.

**Parameters:**
- `year` (int): The year (e.g., 2020)

**Returns:**
- `List[int]`: List of available week numbers, empty list if year not found

**Example:**
```python
from src.data.reader_weekly_offdef import list_available_weeks

# Get list of available weeks for 2020
weeks = list_available_weeks(2020)
print(f"Available weeks: {weeks}")  # e.g., [1, 2, 3, ..., 17]
```

### list_available_file_types(year, week)

Lists all available file types for a specific year and week.

**Parameters:**
- `year` (int): The year (e.g., 2020)
- `week` (int): The week number (e.g., 15)

**Returns:**
- `List[str]`: List of available file types (without .csv extension)

**Example:**
```python
from src.data.reader_weekly_offdef import list_available_file_types

# Get available file types for week 15 of 2020
file_types = list_available_file_types(2020, 15)
print(f"Available files: {file_types}")  # e.g., ['def', 'off', 'kicking']
```

### normalize_stats_by_game(df, column_names, games_column='G', suffix='PerGame')

Generic helper function to create per-game average columns for multiple stats.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `column_names` (List[str]): List of column names to divide by games
- `games_column` (str): Name of the games column (default: 'G')
- `suffix` (str): Suffix to add to the new column names (default: 'PerGame')

**Returns:**
- `pandas.DataFrame`: DataFrame with the new per-game columns added

**Example:**
```python
from src.data.reader_weekly_offdef import read_all_weeks, normalize_stats_by_game

# Read data and normalize multiple stats by games
df = read_all_weeks(2020, 'off')
df_normalized = normalize_stats_by_game(df, ['TotalYds', 'PassYds', 'RushYds'])
# Creates: TotalYdsPerGame, PassYdsPerGame, RushYdsPerGame columns
```

## Directory Structure

The functions expect data to be organized in this structure:

```
data/
├── 2020/
│   ├── week1/
│   │   ├── def.csv
│   │   ├── off.csv
│   │   └── kicking.csv
│   ├── week2/
│   │   ├── def.csv
│   │   ├── off.csv
│   │   └── kicking.csv
│   └── ...
├── 2021/
│   └── ...
└── ...
```

## Usage Examples

```python
from src.data.reader_weekly_offdef import (
    read_weekly_file, 
    read_all_weeks,
    normalize_stats_by_game
)
from src.data.weekly_info import (
    list_available_weeks,
    list_available_file_types
)

# Read specific week's defensive data
week15_def = read_weekly_file(2020, 15, 'def')  # Normalized by default
week15_def_raw = read_weekly_file(2020, 15, 'def', normalize=False)  # Raw data

# Read all defensive data for 2020 season
all_2020_def = read_all_weeks(2020, 'def')

# Check what weeks are available
available_weeks = list_available_weeks(2020)

# Check what file types exist for a specific week
file_types = list_available_file_types(2020, 15)

# Combine multiple file types
def_data = read_all_weeks(2020, 'def')
off_data = read_all_weeks(2020, 'off')

# Read data and normalize multiple stats by games
df = read_all_weeks(2020, 'off')
df_normalized = normalize_stats_by_game(df, ['TotalYds', 'PassYds', 'RushYds'])
# Creates: TotalYdsPerGame, PassYdsPerGame, RushYdsPerGame columns
```

## Error Handling

- All functions include error handling and will print informative messages
- Functions return `None` or empty lists when files/directories are not found
- Individual file read errors don't stop `read_all_weeks()` from processing other files

## Notes

- When using `read_all_weeks()`, the resulting DataFrame includes `week` and `year` columns to track data origin
- File paths are resolved relative to the project root directory
- Week directories should be named `weekX` where X is the week number
- CSV files should be named with the file type as the filename (e.g., `def.csv`, `off.csv`)