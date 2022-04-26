# config.py
import os 
from pathlib import Path

# epochs to train on
EPOCHS = 10

# paths 
HOME_DIR = Path.home()
MAIN_DIR = Path.cwd().parent
RAW_FILE_PATH = MAIN_DIR / "data" / "raw"
TRAIN_FILE_PATH = MAIN_DIR / "data" / "final"
MODEL_OUTPUT_PATH = MAIN_DIR / "models"
NOTEBOOKS_PATH = MAIN_DIR / "notebooks"
REPORTS_PATH = MAIN_DIR / "reports" / "figures"

# colours 
DEFAULT_BLUE = '#4878d0'
DEFAULT_ORANGE = '#ee854a'
DEFAULT_GREEN = '#6acc64'
DEFAULT_RED = '#d65f5f'

# visualisations
DEFAULT_FIGSIZE = (16, 10)
DEFAULT_AXIS_FONT_SIZE = 12
DEFAULT_PLOT_LINESTYLE = ':'
DEFAULT_PLOT_LINEWIDTH = 1
DEFAULT_DASHES = (1,5)

# ML 
RANDOM_STATE = 123
N_JOBS = -1

def main() -> None: 
    pass 

if __name__ == "__main__": 
    main() 