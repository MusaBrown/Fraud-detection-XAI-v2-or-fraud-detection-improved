"""
Download ULB Credit Card Fraud dataset from Kaggle.
Requires Kaggle API: pip install kaggle
"""
import subprocess
from pathlib import Path

def download():
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading ULB Credit Card Fraud dataset from Kaggle...")
    print("Dataset: mlg-ulb/creditcardfraud")
    print("Size: ~150 MB")
    
    try:
        # Download
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "mlg-ulb/creditcardfraud",
            "-p", str(data_dir),
            "--unzip"
        ], check=True)
        
        print("Download complete!")
        print(f"Data location: {data_dir}/creditcard.csv")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure:")
        print("1. Kaggle API is installed: pip install kaggle")
        print("2. Kaggle credentials are in ~/.kaggle/kaggle.json")
        print("   Get credentials from: https://www.kaggle.com/account")
        print("\nAlternative: Download manually from:")
        print("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("And place 'creditcard.csv' in 'data/raw/' folder")

if __name__ == "__main__":
    download()
