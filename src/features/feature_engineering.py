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


def feature_engineering(files: list = None):
    """
    Feature engineering on the data consisting of: 
        1) Imputing missing values 
        2) Vectorizing categorical data 
        3) Binning numerical data 
        4) Dropping columns and rows 
    """

    logger = logging.getLogger()
    logger.info("FEATURE ENGINEERING")

    for file_path in files:
        df = pd.read_parquet(config.INT_FILE_PATH / file_path)

        df.drop(
            columns=[
                "id",
                "year",
                "interest_rate_spread",
                "rate_of_interest",
                "upfront_charges",
                "secured_by",
                "construction_type",
                "security_type",
            ],
            inplace=True,
        )

        df.dropna(
            axis=0,
            how="any",
            subset=[
                "approv_in_adv",
                "loan_purpose",
                "term",
                "neg_ammortization",
                "age",
                "submission_of_application",
            ],
            inplace=True,
        )
        logger.info(f"BINNING NUMERICAL DATA: {file_path}")

        for col in ["property_value", "ltv", "dtir1"]:
            df[f"{col}_binned"] = helpers.bin_values(df, col)
            dummies = pd.get_dummies(df[f"{col}_binned"], prefix=col, prefix_sep="_")
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col, f"{col}_binned", f"{col}_na"], inplace=True)

        logger.info(f"VECTORIZING CATEGORICAL DATA: {file_path}")

        for col in df.columns:
            if not types.is_categorical_dtype(df[col]):
                continue
            if col == config.TARGET:
                continue
            dummies = pd.get_dummies(
                df[col], prefix=col, prefix_sep="_", drop_first=True
            )
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=col, inplace=True)

        logger.info(f"IMPUTING MISSING DATA: {file_path}")

        # impute missing income and loan_limit using the subset of variables
        for col in ["income", "loan_limit"]:
            df_subset = df.loc[
                :,
                [
                    "loan_amount",
                    "credit_worthiness",
                    "credit_score",
                    "interest_only",
                    "loan_purpose_p2",
                    "loan_purpose_p3",
                    "loan_purpose_p4",
                    "loan_type_t2",
                    "loan_type_t3",
                    col,
                ],
            ]
            df_filled = KNNImputer().fit_transform(df_subset)
            df[col] = df_filled[:, -1]

        final_file_path = file_path.replace("init_", "")
        df.to_parquet(config.FIN_FILE_PATH / final_file_path)
        logger.info(f"FILE SAVED AS: {final_file_path}")

    logger.info("DONE EXPORTS")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Read in files to perform feature engineering on"
    )
    parser.add_argument("--files", nargs="*", default=[])
    args = parser.parse_args()

    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    feature_engineering(files=args.files)
