#%%
import os
import shutil
import re
from pathlib import Path
import pandas as pd



#%%
exp_name = '2026_02_Alpha3_dendrites' # Update this path accordingly
day_name = '2026_02_25' # Update this path accordingly

main_dir = "/Volumes/tungsten/scratch/gfelsenb/Ana/2p-imaging/burak/"
exp_path = os.path.join(main_dir, exp_name) # Update this path accordingly
day_folder_path = os.path.join(exp_path, day_name) # Update this path accordingly

day_folder = Path(day_folder_path)

if not day_folder.exists():
    print(f"Error: Folder {day_folder_path} does not exist")
    exit('Terminating script.')

# Pattern to match S1-T##### at the beginning of filename
pattern = re.compile(r'^(S1-T\d{5})')

# Valid extensions to process
valid_extensions = {'.osf', '.bin', '.ini', '.ach', '.lvd'}

# Get all files in the day folder
files = [f for f in day_folder.iterdir() if f.is_file()]

for file in files:
    # Check if file has valid extension
    if file.suffix.lower() not in valid_extensions:
        print(f"Skipping {file.name}: Invalid extension {file.suffix}")
        continue
    
    # Extract identifier from filename
    match = pattern.match(file.name)
    
    if match:
        identifier = match.group(1)
        target_folder = day_folder / identifier
        
        # Check if target folder exists
        if target_folder.exists() and target_folder.is_dir():
            target_path = target_folder / file.name
            
            try:
                shutil.move(str(file), str(target_path))
                print(f"Moved: {file.name} -> {identifier}/")
            except Exception as e:
                print(f"Error moving {file.name}: {e}")
        else:
            print(f"Warning: Target folder {identifier} does not exist for {file.name}")
    else:
        print(f"Warning: No valid identifier found in {file.name}")

#%%
# Update database with new seriesIDs
print("\nUpdating database with new seriesIDs...")

# Find the database CSV file
database_files = list(Path(exp_path).glob("*_database.csv"))
database_files = [f for f in database_files if not f.name.startswith('.')]

if not database_files:
    print(f"Error: No database CSV file found in {exp_path}")
else:
    if len(database_files) > 1:
        print(f"Warning: Multiple database files found. Using {database_files[0].name}")
    
    database_path = database_files[0]
    
    # Read existing database
    df = pd.read_csv(database_path)
    
    # Get existing seriesIDs
    if 'seriesID' not in df.columns:
        print("Error: 'seriesID' column not found in database")
    else:
        existing_series = set(df['seriesID'].dropna().astype(str))
        
        # Pattern to match S1-T##### folder names
        series_pattern = re.compile(r'^S1-T\d{5}$')
        
        # Find all series folders in the day folder
        series_folders = [f for f in day_folder.iterdir() 
                         if f.is_dir() and series_pattern.match(f.name)]
        
        # Find new seriesIDs
        new_series = []
        for folder in series_folders:
            series_id = folder.name
            if series_id not in existing_series:
                new_series.append(series_id)
        
        if not new_series:
            print("No new seriesIDs found to add to database")
        else:
            # Create new rows for the new seriesIDs
            new_rows = pd.DataFrame({'seriesID': new_series})
            
            # Append to existing dataframe
            df_updated = pd.concat([df, new_rows], ignore_index=True)
            
            # Save updated database
            df_updated.to_csv(database_path, index=False)
            
            print(f"Added {len(new_series)} new seriesIDs to database:")
            for series_id in new_series:
                print(f"  - {series_id}")
#%%