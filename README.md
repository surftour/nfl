# NFL Data Analysis Project

A Python project for analyzing NFL data using ESPN APIs and weekly statistics.


## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For Jupyter notebook support:
```bash
jupyter notebook
# or
jupyter lab
```

## Code Quality

This project uses pylint for code quality checks:

```bash
# Run pylint checks
./lint.sh

# Or run manually
pylint src/
```

The pylint configuration (`.pylintrc`) is customized for practical Python development with reasonable complexity limits and disabled overly strict style checks.

## Available Functions

### ESPN API Functions (`src/api/espn.py`)
- `fetch_teams()` - Fetches all NFL team information
- `fetch_team_stats(year, team_id)` - Fetches team statistics
- `fetch_game_results(year, week, season_type)` - Fetches game results

### Weekly Data Reader (`src/data/reader_weekly_offdef.py`)
- `read_weekly_file(year, week, file_type)` - Reads specific weekly CSV files
- `read_all_weeks(year, file_type)` - Combines all weekly files for a year
- Automatically handles column name concatenation and data type inference
- Separates league totals/averages from team data

### Weekly Results Reader (`src/data/reader_weekly_results.py`)
- `read_weekly_game_results(file_path)` - Reads TSV files containing weekly game results
- Returns DataFrame with columns: date, visiting_team, visiting_score, home_team, home_score, status
- Parses game data where each row represents a single NFL game

### Utility Functions (`src/utils/data_io.py`)
- `save_to_csv(fetch_function, filename, *args, **kwargs)` - Generic CSV export

## Usage Examples

```python
from src.api.espn import fetch_teams, fetch_team_stats
from src.data.reader_weekly_offdef import read_weekly_file, read_all_weeks
from src.data.reader_weekly_results import read_weekly_game_results
from src.utils.data_io import save_to_csv

# Fetch and save team data
teams_df = save_to_csv(fetch_teams, 'data/nfl_teams.csv')

# Read weekly defensive stats
def_data = read_weekly_file(2020, 15, 'def')

# Read all defensive data for 2020 season
all_def_data = read_all_weeks(2020, 'def')

# Read weekly game results from TSV file
games_df = read_weekly_game_results('data/2023/week14/results.tsv')
```

## Dependencies

- `requests>=2.31.0` - HTTP requests to ESPN APIs
- `pandas>=2.0.0` - Data manipulation and CSV export
- `jupyter>=1.0.0` - Jupyter notebook support
- `notebook>=6.0.0` - Classic Jupyter Notebook interface
- `jupyterlab>=3.0.0` - Modern JupyterLab interface
- `pylint>=2.17.0` - Code quality checks

## Data Format

Weekly CSV files are expected in the format:
```
data/YYYY/weekNN/filename.csv
```

Weekly game results TSV files are expected in the format:
```
data/YYYY/weekNN/results.tsv
```

The reader automatically:
- Concatenates first two rows as column names
- Removes spaces and replaces special characters in column names
- Infers data types (integers vs floats vs strings)
- Adds year/week columns based on file path
- Separates league totals from team data

## Development

For detailed development instructions and function documentation, see `CLAUDE.md`.