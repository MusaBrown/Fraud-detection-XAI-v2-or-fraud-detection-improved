"""
Download IEEE-CIS Fraud Detection dataset from Kaggle.
Requires Kaggle API credentials (~/.kaggle/kaggle.json).
"""
import os
import subprocess
from pathlib import Path

def download_ieee_cis():
    """Download IEEE-CIS dataset from Kaggle."""
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("WARNING: IEEE-CIS data was NOT used in this project.")
    print("Downloading IEEE-CIS Fraud Detection dataset from Kaggle...")
    print("Note: This requires Kaggle API credentials.")
    print("File size: ~1.5 GB")
    
    try:
        # Download using kaggle API
        subprocess.run([
            "kaggle", "competitions", "download",
            "-c", "ieee-fraud-detection",
            "-p", str(data_dir)
        ], check=True)
        
        # Unzip
        import zipfile
        zip_path = data_dir / "ieee-fraud-detection.zip"
        if zip_path.exists():
            print("Extracting files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            zip_path.unlink()
            print("Download complete!")
            
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("\nPlease ensure:")
        print("1. Kaggle API is installed: pip install kaggle")
        print("2. Kaggle credentials are in ~/.kaggle/kaggle.json")
        print("   (Get from https://www.kaggle.com/account)")
    except FileNotFoundError:
        print("Kaggle API not found. Install with: pip install kaggle")

if __name__ == "__main__":
    download_ieee_cis()
