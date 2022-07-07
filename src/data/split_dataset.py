import logging
import os
import warnings

import pandas as pd
from src import config
from src.data.make_dataset import make_dataset

warnings.filterwarnings("ignore")


def split_dataset(threshold: int = 300_000):

    logger = logging.getLogger()
    logger.info(f"SPLITTING LOAN SIZE BY THRESHOLD: {threshold}")

    if not os.path.exists(config.INT_FILE_PATH / config.BASIC_CLEAN_FILE_NAME):
        make_dataset()

    df = pd.read_parquet(config.INT_FILE_PATH / config.BASIC_CLEAN_FILE_NAME)

    # split into large and small loans
    df_small = df.loc[df["loan_amount"] < threshold, :]
    df_large = df.loc[df["loan_amount"] >= threshold, :]

    df_small.to_parquet(config.INT_FILE_PATH / f"df_small_loans_{threshold}.parquet")
    df_large.to_parquet(config.INT_FILE_PATH / f"df_large_loans_{threshold}.parquet")

    logger.info(f"THRESHOLD SPLIT AT {threshold} DONE")
    logger.info(
        f"SMALL LOANS PATH: {config.INT_FILE_PATH / f'df_small_loans_{threshold}.parquet'}"
    )
    logger.info(
        f"LARGE LOANS PATH: {config.INT_FILE_PATH / f'df_large_loans_{threshold}.parquet'}"
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    import argparse

    from src.helpers import restricted_int

    parser = argparse.ArgumentParser(
        description="Process threshold required for small-large split"
    )
    parser.add_argument(
        "--threshold", type=restricted_int, default=300_000, required=True
    )
    args = parser.parse_args()

    split_dataset(threshold=args.threshold)
