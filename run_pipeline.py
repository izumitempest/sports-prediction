import os
import subprocess
import sys

def run_script(script_name, args=[]):
    print(f"--- Running {script_name} ---")
    cmd = [sys.executable, script_name] + args
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Error running {script_name}")
        return False
    return True

if __name__ == "__main__":
    # 1. Download Data & Preprocess
    if not run_script(os.path.join("src", "data_loader.py")):
        exit(1)
        
    # 2. Train Model (Random Forest)
    if not run_script(os.path.join("src", "train_model.py")):
        exit(1)
        
    print("\nPipeline finished successfully.")
    print("Run predictions with: python src/predict.py 'HomeTeam' 'AwayTeam'")
