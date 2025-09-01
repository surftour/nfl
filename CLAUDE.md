# NFL Data Analysis Project

## Project Structure

```
/Users/tj/nfl/
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── espn.py          # ESPN API functions
│   └── utils/
│       ├── __init__.py
│       └── data_io.py       # Generic save_to_csv function
├── data/                    # For CSV outputs
├── notebooks/               # For analysis notebooks
├── requirements.txt         # Dependencies
└── CLAUDE.md               # This file
```

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Available Functions

### ESPN API Functions (`src/api/espn.py`)

See detailed documentation in `src/api/CLAUDE.md`

1. **`fetch_teams()`** - Fetches all NFL team information
2. **`fetch_team_stats(year, team_id)`** - Fetches team statistics for a specific year and team  
3. **`fetch_game_results(year, week=None, season_type=2)`** - Fetches game results for a specific year and week

### Utility Functions (`src/utils/data_io.py`)

1. **`save_to_csv(fetch_function, filename, *args, **kwargs)`**
   - Generic function to call any fetch function and save results to CSV
   - Parameters:
     - `fetch_function`: The function to call (e.g., fetch_teams, fetch_team_stats)
     - `filename`: Name of the CSV file to save to
     - `*args, **kwargs`: Arguments to pass to the fetch function
   - Returns DataFrame if successful, None if failed

## Usage Examples

```python
from src.api.espn import fetch_teams, fetch_team_stats, fetch_game_results
from src.utils.data_io import save_to_csv

# Fetch all NFL teams and save to CSV
teams_df = save_to_csv(fetch_teams, 'data/nfl_teams.csv')

# Fetch team stats for Philadelphia Eagles (ID: 25) in 2024
stats_df = save_to_csv(fetch_team_stats, 'data/eagles_2024_stats.csv', 2024, 25)

# Fetch Week 1 game results for 2024
games_df = save_to_csv(fetch_game_results, 'data/week1_games.csv', 2024, 1)

# Direct function calls
teams = fetch_teams()
eagles_stats = fetch_team_stats(2024, 25)
week1_games = fetch_game_results(2024, week=1)
```

## Running the Module

```bash
# Run ESPN module directly (saves sample data to data/ directory)
python -m src.api.espn
```

## Dependencies

- `requests>=2.31.0` - HTTP requests to ESPN APIs
- `pandas>=2.0.0` - Data manipulation and CSV export

## Notes

- All CSV files are saved to the `data/` directory by default
- Team IDs can be found by first calling `fetch_teams()` to get the mapping
- The project uses ESPN's public APIs - no authentication required
- Season type 2 refers to regular season statistics