# Pro Football Reference API Functions Documentation

This module uses the **nflscraPy** library to scrape NFL data from Pro Football Reference.

**Important Notes:**
- Pro Football Reference does not have a public API - this module uses web scraping
- Rate limiting is built-in (3.5-5.5 second delays between requests)
- Data is available from 1970 onwards
- Respect the server - avoid excessive requests

## fetch_season_schedule(season)

Fetches all games for a season from Pro Football Reference.

### Parameters
- `season` (int): Season year (e.g., 2024). Data available from 1970 onwards.

### Returns
pandas.DataFrame with game information including:
- **season**: Season year
- **week**: Week number
- **game_day_of_week**: Day of week (e.g., "Sunday")
- **game_date**: Date of game
- **boxscore_word**: Game type (e.g., "Final", "Playoffs")
- **winning_team**: Winning team name
- **winning_team_points**: Points scored by winner
- **losing_team**: Losing team name
- **losing_team_points**: Points scored by loser
- **boxscore_stats_link**: URL to detailed boxscore (key for other functions)
- **home_team**: Home team indicator
- **winning_team_abbr**: Winning team abbreviation
- **losing_team_abbr**: Losing team abbreviation

### Data Source
- URL Pattern: `https://www.pro-football-reference.com/years/{season}/games.htm`
- Underlying function: `nflscraPy._gamelogs(season)`

### Usage Example
```python
from src.api.pro_football_reference import fetch_season_schedule
from src.utils.data_io import save_to_csv

# Fetch all games for 2024 season
schedule = fetch_season_schedule(2024)

# Save to CSV
schedule_df = save_to_csv(fetch_season_schedule, 'data/pfr_2024_schedule.csv', 2024)

# Explore the data
print(schedule[['week', 'game_date', 'winning_team', 'losing_team', 'winning_team_points', 'losing_team_points']].head())

# Get boxscore URLs for detailed data
boxscore_urls = schedule['boxscore_stats_link'].tolist()
```

---

## fetch_game_metadata(boxscore_url)

Fetches game metadata (venue, weather, betting lines, etc.) for a specific game.

### Parameters
- `boxscore_url` (str): Boxscore URL from `fetch_season_schedule`'s `boxscore_stats_link` column
  - Format: `https://www.pro-football-reference.com/boxscores/YYYYMMDDTEAM.htm`
  - Example: `https://www.pro-football-reference.com/boxscores/202212180jax.htm`

### Returns
pandas.DataFrame with metadata including:
- **boxscore_stats_link**: Game identifier URL
- **vegas_line**: Betting line (e.g., "Jaguars -3.0")
- **over_under**: Over/under line
- **roof**: Roof type ("outdoors", "dome", "retractable roof", etc.)
- **surface**: Playing surface ("grass", "fieldturf", etc.)
- **duration**: Game duration (e.g., "3:05")
- **attendance**: Attendance count
- **weather**: Weather conditions (temperature, wind, humidity)
- **start_time**: Game start time

### Data Source
- URL Pattern: Boxscore URL from season schedule
- Underlying function: `nflscraPy._gamelog_metadata(boxscore_url)`
- Note: Some metadata is extracted from HTML comments

### Usage Example
```python
from src.api.pro_football_reference import fetch_season_schedule, fetch_game_metadata

# First, get the season schedule
schedule = fetch_season_schedule(2024)

# Get metadata for first game
boxscore_url = schedule.iloc[0]['boxscore_stats_link']
metadata = fetch_game_metadata(boxscore_url)

# View metadata
print(metadata[['roof', 'surface', 'weather', 'attendance']])

# Batch process multiple games
metadata_list = []
for url in schedule.head(5)['boxscore_stats_link']:
    meta = fetch_game_metadata(url)
    metadata_list.append(meta)

all_metadata = pd.concat(metadata_list, ignore_index=True)
```

---

## fetch_game_statistics(boxscore_url)

Fetches game statistics (team stats, scoring, turnovers) for a specific game.

### Parameters
- `boxscore_url` (str): Boxscore URL from `fetch_season_schedule`'s `boxscore_stats_link` column
  - Format: `https://www.pro-football-reference.com/boxscores/YYYYMMDDTEAM.htm`

### Returns
pandas.DataFrame with statistics (typically 2 rows - one per team):
- **boxscore_stats_link**: Game identifier URL
- **team**: Team abbreviation
- **first_downs**: First downs
- **rush_yards**: Rushing yards
- **rush_att**: Rushing attempts
- **pass_yards**: Passing yards (gross)
- **pass_cmp**: Passes completed
- **pass_att**: Passes attempted
- **sacks**: Sacks allowed
- **net_pass_yards**: Net passing yards (after sacks)
- **total_yards**: Total yards
- **fumbles**: Fumbles
- **fumbles_lost**: Fumbles lost
- **turnovers**: Total turnovers
- **penalties**: Penalties
- **penalty_yards**: Penalty yards
- **third_down_conversions**: Third down conversions (e.g., "5-12")
- **fourth_down_conversions**: Fourth down conversions (e.g., "1-2")
- **time_of_possession**: Time of possession (e.g., "32:15")

### Data Source
- URL Pattern: Boxscore URL from season schedule
- Underlying function: `nflscraPy._gamelog_statistics(boxscore_url)`

### Usage Example
```python
from src.api.pro_football_reference import fetch_season_schedule, fetch_game_statistics
import pandas as pd

# Get season schedule
schedule = fetch_season_schedule(2024)

# Get statistics for first game
boxscore_url = schedule.iloc[0]['boxscore_stats_link']
stats = fetch_game_statistics(boxscore_url)

# View stats
print(stats[['team', 'total_yards', 'turnovers', 'time_of_possession']])

# Compare teams in a game
home_stats = stats[stats['team'] == 'JAX']
away_stats = stats[stats['team'] == 'IND']
```

---

## fetch_season_with_details(season, include_metadata=True, include_statistics=True, max_games=None)

Batch processing function that fetches season schedule and detailed data for all (or subset of) games.

### Parameters
- `season` (int): Season year (e.g., 2024)
- `include_metadata` (bool): Whether to fetch metadata for each game (default: True)
- `include_statistics` (bool): Whether to fetch statistics for each game (default: True)
- `max_games` (int, optional): Maximum number of games to process (useful for testing)

### Returns
Dictionary with three keys:
- **'schedule'**: DataFrame from `fetch_season_schedule`
- **'metadata'**: DataFrame with metadata for all games (if `include_metadata=True`)
- **'statistics'**: DataFrame with statistics for all games (if `include_statistics=True`)

### Performance Notes
⚠️ **WARNING**: This function makes many HTTP requests with built-in rate limiting:
- Each game requires 1-2 requests (depending on parameters)
- Rate limiting: 3.5-5.5 seconds between requests
- Full season (~280 games): **30-45 minutes** total processing time
- Use `max_games` parameter to test with a subset first

### Usage Example
```python
from src.api.pro_football_reference import fetch_season_with_details

# Test with first 5 games (~30 seconds)
print("Testing with 5 games...")
test_data = fetch_season_with_details(2024, max_games=5)

print(f"Schedule: {len(test_data['schedule'])} games")
print(f"Metadata: {len(test_data['metadata'])} records")
print(f"Statistics: {len(test_data['statistics'])} records")

# Save test data
test_data['schedule'].to_csv('data/pfr_test_schedule.csv', index=False)
test_data['metadata'].to_csv('data/pfr_test_metadata.csv', index=False)
test_data['statistics'].to_csv('data/pfr_test_statistics.csv', index=False)

# Fetch full season (WARNING: 30-45 minutes!)
print("\nFetching full season...")
full_data = fetch_season_with_details(2024)

# Save full season data
full_data['schedule'].to_csv('data/pfr_2024_schedule_full.csv', index=False)
full_data['metadata'].to_csv('data/pfr_2024_metadata_full.csv', index=False)
full_data['statistics'].to_csv('data/pfr_2024_statistics_full.csv', index=False)
```

---

## Complete Workflow Example

### Step 1: Get Season Schedule
```python
from src.api.pro_football_reference import fetch_season_schedule
from src.utils.data_io import save_to_csv

# Fetch all games for 2024
schedule = save_to_csv(fetch_season_schedule, 'data/pfr_2024_schedule.csv', 2024)

print(f"Found {len(schedule)} games")
print(schedule[['week', 'game_date', 'winning_team', 'losing_team']].head(10))
```

### Step 2: Get Detailed Data for Specific Games
```python
from src.api.pro_football_reference import fetch_game_metadata, fetch_game_statistics

# Filter for Week 1 games
week1 = schedule[schedule['week'] == 1]

# Get details for each Week 1 game
for idx, game in week1.iterrows():
    url = game['boxscore_stats_link']

    # Fetch metadata and stats
    metadata = fetch_game_metadata(url)
    stats = fetch_game_statistics(url)

    print(f"\n{game['winning_team']} vs {game['losing_team']}")
    print(f"Weather: {metadata.iloc[0]['weather']}")
    print(f"Surface: {metadata.iloc[0]['surface']}")
    print(f"Total yards: {stats['total_yards'].tolist()}")
```

### Step 3: Batch Processing with Testing
```python
from src.api.pro_football_reference import fetch_season_with_details

# Step 3a: Test with small subset
print("Testing with 3 games...")
test = fetch_season_with_details(2024, max_games=3)

# Verify data looks good
assert not test['schedule'].empty
assert not test['metadata'].empty
assert not test['statistics'].empty

# Step 3b: Process full season (if test successful)
print("\nProcessing full season (this will take 30-45 minutes)...")
full = fetch_season_with_details(2024)

# Save all data
full['schedule'].to_csv('data/pfr_2024_complete_schedule.csv', index=False)
full['metadata'].to_csv('data/pfr_2024_complete_metadata.csv', index=False)
full['statistics'].to_csv('data/pfr_2024_complete_statistics.csv', index=False)

print("Complete!")
```

---

## Combining with ESPN Data

You can combine Pro Football Reference data with ESPN API data for comprehensive analysis:

```python
from src.api.pro_football_reference import fetch_season_schedule
from src.api.espn import fetch_game_results
import pandas as pd

# Get data from both sources
pfr_schedule = fetch_season_schedule(2024)
espn_schedule = fetch_game_results(2024)

# PFR provides: weather, betting lines, detailed team stats
# ESPN provides: play-by-play data, real-time scores, additional metadata

# Join on game date and teams for comprehensive dataset
# (requires team name standardization)
```

---

## Data Availability

### Season Schedule
- Available: 1970 - present
- Updates: After each game concludes

### Game Metadata & Statistics
- Available: 1970 - present
- Includes: Regular season, playoffs, Super Bowl
- Weather data: Available for outdoor stadiums (limited for domes)

---

## Rate Limiting and Best Practices

### Built-in Rate Limiting
- Random delay: 3.5-5.5 seconds between requests
- Automatically handled by nflscraPy
- **DO NOT** modify or remove rate limiting

### Best Practices
1. **Test first**: Use `max_games` parameter before full season fetch
2. **Cache results**: Save CSVs immediately to avoid re-fetching
3. **Be patient**: Full season processing takes 30-45 minutes
4. **Schedule wisely**: Run during off-peak hours
5. **Handle errors**: Check for empty DataFrames

### Respectful Usage
```python
# ✅ GOOD: Test with subset first
test = fetch_season_with_details(2024, max_games=5)
if not test['schedule'].empty:
    full = fetch_season_with_details(2024)

# ❌ BAD: No rate limiting bypass
# Don't modify nflscraPy's sleep intervals

# ✅ GOOD: Cache results
schedule = fetch_season_schedule(2024)
schedule.to_csv('data/cached_schedule.csv', index=False)

# ❌ BAD: Repeated fetches
for i in range(10):
    schedule = fetch_season_schedule(2024)  # Wasteful!
```

---

## Troubleshooting

### Issue: "nflscraPy library not installed"
```bash
pip install nflscraPy beautifulsoup4
```

### Issue: Empty DataFrame returned
- Check season year (must be >= 1970)
- Check internet connection
- Check if Pro Football Reference is accessible
- Verify boxscore_url format

### Issue: Processing is slow
- This is expected due to rate limiting
- Use `max_games` parameter for testing
- Consider processing in batches (e.g., by week)

### Issue: Data columns missing
- Different seasons may have different available data
- Check Pro Football Reference website for data availability
- Some fields (e.g., weather) may be missing for dome games

---

## Data Schema Reference

### Season Schedule Columns
```
season               int64
week                 int64
game_day_of_week     object
game_date            object
boxscore_word        object
winning_team         object
winning_team_points  int64
losing_team          object
losing_team_points   int64
boxscore_stats_link  object
home_team            object
winning_team_abbr    object
losing_team_abbr     object
```

### Game Metadata Columns
```
boxscore_stats_link  object
vegas_line           object
over_under           float64
roof                 object
surface              object
duration             object
attendance           int64
weather              object
start_time           object
```

### Game Statistics Columns
```
boxscore_stats_link        object
team                       object
first_downs                int64
rush_yards                 int64
rush_att                   int64
pass_yards                 int64
pass_cmp                   int64
pass_att                   int64
sacks                      int64
net_pass_yards             int64
total_yards                int64
fumbles                    int64
fumbles_lost               int64
turnovers                  int64
penalties                  int64
penalty_yards              int64
third_down_conversions     object
fourth_down_conversions    object
time_of_possession         object
```
