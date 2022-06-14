# config.py
import os
from pathlib import Path

from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import ShuffleSplit

# epochs to train on
EPOCHS = 10

# paths
HOME_DIR = Path().home()
ROOT_DIR = Path(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
RAW_FILE_PATH = ROOT_DIR / "data" / "raw"
INT_FILE_PATH = ROOT_DIR / "data" / "interim"
FIN_FILE_PATH = ROOT_DIR / "data" / "final"
MODEL_OUTPUT_PATH = ROOT_DIR / "models"
NOTEBOOKS_PATH = ROOT_DIR / "notebooks"
REPORTS_PATH = ROOT_DIR / "reports" / "figures"

# colours
DEFAULT_BLUE = "#4878d0"
DEFAULT_ORANGE = "#ee854a"
DEFAULT_GREEN = "#6acc64"
DEFAULT_RED = "#d65f5f"

# filenames
RAW_FILE_NAME = "Loan_Default.csv"
INT_FILE_NAME = "df.parquet"


# visualisations
DEFAULT_FIGSIZE = (16, 10)
DEFAULT_AXIS_FONT_SIZE = 12
DEFAULT_PLOT_LINESTYLE = ":"
DEFAULT_PLOT_LINEWIDTH = 1
DEFAULT_DASHES = (1, 5)

# ML
RANDOM_STATE = 123
N_JOBS = -1
N_SPLITS = 5
TARGET = "status"
SMALL_SCORER = {"AUC": "roc_auc", "F_score": make_scorer(fbeta_score, beta=0.5)}
LARGE_SCORER = {"AUC": "roc_auc", "F_score": make_scorer(fbeta_score, beta=2)}
CV_SPLIT = ShuffleSplit(
    n_splits=5, test_size=0.2, train_size=0.8, random_state=RANDOM_STATE
)


def main() -> None:
    pass


if __name__ == "__main__":
    main()

