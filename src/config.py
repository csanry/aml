# config.py
import os 
from pathlib import Path

# epochs to train on
EPOCHS = 10

# paths 
HOME_DIR = Path.home()
TRAIN_FILE_PATH = HOME_DIR / "data" / "final"
MODEL_OUTPUT_PATH = HOME_DIR / "models"
NOTEBOOKS_PATH = HOME_DIR / "notebooks"
REPORTS_PATH = HOME_DIR / "reports" / "figures"


def main() -> None: 
    pass 

if __name__ == "__main__": 
    main() 