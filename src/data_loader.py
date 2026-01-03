import os
import requests
import pandas as pd

DATA_DIR = os.path.join("data", "raw")
BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Seasons to download (e.g., 2019/2020 to 2023/2024)
# Format is '1920', '2021', etc.
SEASONS = ['1920', '2021', '2122', '2223', '2324']
LEAGUE = "E0" # Premier League

def download_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    all_data = []

    for season in SEASONS:
        url = f"{BASE_URL}/{season}/{LEAGUE}.csv"
        # Check if file exists to avoid re-downloading if not needed, 
        # but for now we'll overwrite to be safe.
        file_path = os.path.join(DATA_DIR, f"{LEAGUE}_{season}.csv")
        
        if not os.path.exists(file_path):
            print(f"Downloading {url}...")
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                print(f"Failed to download {season}: {e}")
                continue
        else:
            print(f"File {file_path} already exists.")
            
        # Verify we can read it
        try:
            df = pd.read_csv(file_path, encoding='unicode_escape') 
            df['Season'] = season
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        full_path = os.path.join(DATA_DIR, "all_seasons.csv")
        full_df.to_csv(full_path, index=False)
        print(f"Combined data saved to {full_path}. Shape: {full_df.shape}")
    else:
        print("No data loaded.")

if __name__ == "__main__":
    download_data()
