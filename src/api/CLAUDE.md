# ESPN API Functions Documentation

## fetch_teams()

Fetches all NFL team information from ESPN API.

### Parameters
- None

### Returns
- pandas DataFrame with team details including:
  - Team ID, UID, slug
  - Team names (display name, short display name, name, nickname, location)
  - Team abbreviation
  - Team colors (primary and alternate)
  - Activity status (isActive, isAllStar)
  - Logo URL

### API Endpoint
- URL: `http://site.api.espn.com/apis/site/v2/sports/football/nfl/teams`

### Usage Example
```python
from src.api.espn import fetch_teams
from src.utils.data_io import save_to_csv

# Fetch all NFL teams
teams = fetch_teams()

# Save to CSV
teams_df = save_to_csv(fetch_teams, 'data/nfl_teams.csv')
```

## fetch_team_stats(year, team_id)

Fetches team statistics for a specific year and team.

### Parameters
- `year` (int): Season year (e.g., 2024)
- `team_id` (int): ESPN team ID (e.g., 25 for Philadelphia Eagles)

### Returns
- pandas DataFrame with statistical categories and values including:
  - Category name
  - Stat name, display name, short display name
  - Stat description
  - Stat value and display value

### API Endpoint
- URL: `https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{year}/types/2/teams/{team_id}/statistics`

### Usage Example
```python
from src.api.espn import fetch_team_stats
from src.utils.data_io import save_to_csv

# Fetch Eagles stats for 2024
eagles_stats = fetch_team_stats(2024, 25)

# Save to CSV
stats_df = save_to_csv(fetch_team_stats, 'data/eagles_2024_stats.csv', 2024, 25)
```

## fetch_game_results(year, week=None, season_type=2)

Fetches NFL game results from ESPN's scoreboard API for a specific year and optionally a specific week.

### Parameters
- `year` (int): Season year (e.g., 2024)
- `week` (int, optional): Week number (1-18 for regular season). If not provided, returns all games for the season
- `season_type` (int, optional): Season type
  - 1 = Preseason
  - 2 = Regular season (default)
  - 3 = Postseason

### Returns
- pandas DataFrame with game information including:
  - Game metadata (ID, date, name, status, completion status)
  - Season and week information
  - Home team details (ID, name, abbreviation, score, winner status)
  - Away team details (ID, name, abbreviation, score, winner status)
  - Venue information (name, city, state)

### API Endpoint
- Base URL: `https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard`
- Parameters: `seasontype`, `week` (optional)

### Usage Example
```python
from src.api.espn import fetch_game_results
from src.utils.data_io import save_to_csv

# Fetch all games for 2024 season
all_games = fetch_game_results(2024)

# Fetch games for a specific week
week1_games = fetch_game_results(2024, week=1)

# Fetch postseason games
playoff_games = fetch_game_results(2024, season_type=3)

# Save to CSV using utility function
games_df = save_to_csv(fetch_game_results, 'data/week1_games.csv', 2024, 1)
```

## General Usage Examples

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

## Error Handling
- All functions return `None` if request fails or JSON parsing errors occur
- Error messages are printed to console for debugging

## Notes
- Team IDs can be found by first calling `fetch_teams()` to get the mapping
- The project uses ESPN's public APIs - no authentication required
- Season type 2 refers to regular season statistics