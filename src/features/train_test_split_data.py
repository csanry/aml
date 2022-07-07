import logging
import os
import warnings

import pandas as pd
import pandas.api.types as types
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from src import config, helpers
from src.data import split_dataset

warnings.filterwarnings("ignore")


def train_test_split_data(files: list = None):

    for file_path in files:
        df = pd.read_parquet(config.INT_FILE_PATH / file_path)

        train_file_path = f"train_init_{file_path}"
        test_file_path = f"test_init_{file_path}"

        train, test = train_test_split(
            df, test_size=0.2, random_state=config.RANDOM_STATE
        )

        train.to_parquet(config.INT_FILE_PATH / train_file_path)
        test.to_parquet(config.INT_FILE_PATH / test_file_path)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Read in files to perform the train test split on"
    )

    parser.add_argument("--files", nargs="*", default=[])
    args = parser.parse_args()

    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    train_test_split_data(files=args.files)
