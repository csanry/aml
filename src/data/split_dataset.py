import logging
import os
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from src.features import build_features
from src import config

warnings.filterwarnings("ignore")


def main(test_size: float = 0.2):

    logger = logging.getLogger()
    logger.info("TRAIN TEST SPLIT")

    if not os.path.exists(
        config.INT_FILE_PATH / "df_large_engineered.parquet"
    ) or not os.path.exists(config.INT_FILE_PATH / "df_small_engineered.parquet"):
        build_features.main()

    df_small = pd.read_parquet(config.INT_FILE_PATH / "df_small_engineered.parquet")
    df_large = pd.read_parquet(config.INT_FILE_PATH / "df_large_engineered.parquet")

    for name, df in zip(["df_small", "df_large"], [df_small, df_large]):
        train, test = train_test_split(
                df, test_size=test_size, random_state=config.RANDOM_STATE
            )

        for df, file_type in zip([train, test], ["train", "test"]):
            df.to_parquet(config.FIN_FILE_PATH / f"{name}_{file_type}.parquet")
    
    logger.info(f"TRAIN TEST SPLIT DONE")


if __name__ == "__main__":
    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()