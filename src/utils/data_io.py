import pandas as pd

def save_to_csv(fetch_function, filename, *args, **kwargs):
    """
    Generic function to call a fetch function and save results to CSV
    
    Args:
        fetch_function: The function to call (e.g., fetch_teams, fetch_team_stats)
        filename: Name of the CSV file to save to
        *args: Arguments to pass to the fetch function
        **kwargs: Keyword arguments to pass to the fetch function
    
    Returns:
        DataFrame if successful, None if failed
    """
    try:
        df = fetch_function(*args, **kwargs)
        
        if df is not None and not df.empty:
            df.to_csv(filename, index=False)
            print(f"Data saved to '{filename}' ({len(df)} rows)")
            return df
        else:
            print("No data to save")
            return None
            
    except Exception as e:
        print(f"Error saving data: {e}")
        return None