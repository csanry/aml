import logging
import os
import warnings
import pandas as pd
from src.data import make_dataset
from src import config

warnings.filterwarnings("ignore")


def main(threshold: int = 300000):

    logger = logging.getLogger()
    logger.info(f"LOAN SIZE SPLIT BY THRESHOLD {threshold}")

    if not os.path.exists(config.INT_FILE_PATH / "df_processed.parquet"):
        make_dataset.main()

    df = pd.read_parquet(config.INT_FILE_PATH / "df_processed.parquet")

    # split into large and small loans
    df_small = df.loc[df["loan_amount"] < threshold, :]
    df_large = df.loc[df["loan_amount"] >= threshold, :]

    df_small.to_parquet(config.INT_FILE_PATH / "df_small_loans.parquet")
    df_large.to_parquet(config.INT_FILE_PATH / "df_large_loans.parquet")

    logger.info(f"THRESHOLD SPLIT AT {threshold} DONE")


if __name__ == "__main__":
    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    import argparse

    def restricted_int(x):
        try:
            x = int(x)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{x} not an integer")

        if x < 0:
            raise argparse.ArgumentTypeError(f"{x} loan value threshold cannot be negative.")
        return x    

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--threshold', type=restricted_int, default=300000, required=False)
    args = parser.parse_args()

    main(args.threshold)