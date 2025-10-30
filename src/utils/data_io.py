import os


def ensure_directory_exists(filepath):
    """Create directory for the filepath if it doesn't exist"""
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def version_existing_file(filepath):
    """
    Version an existing file if it exists by renaming it with .v1, .v2, etc.

    Args:
        filepath: Full path to the file to version

    Returns:
        versioned_path if file was versioned, None if file didn't exist
    """
    if not os.path.exists(filepath):
        return None

    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    name_without_ext, ext = os.path.splitext(filename)

    version = 1
    while os.path.exists(f"{directory}/{name_without_ext}.v{version}{ext}"):
        version += 1

    versioned_path = f"{directory}/{name_without_ext}.v{version}{ext}"
    os.rename(filepath, versioned_path)
    print(f"Moved existing file to {versioned_path}")
    return versioned_path


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

        print("No data to save")
        return None

    except Exception as e:
        print(f"Error saving data: {e}")
        return None
