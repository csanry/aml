import logging
import os
import warnings

import numpy as np
import pandas as pd
import pandas.api.types as types
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from src import config, helpers
from src.data import make_dataset

warnings.filterwarnings("ignore")


def main():
    """Feature engineering on the data consisting of 
    Imputing missing values 
    Dropping columns and rows 
    
    """

    logger = logging.getLogger()
    logger.info("FEATURE ENGINEERING")

    if not os.path.exists(config.INT_FILE_PATH / config.INT_FILE_NAME):
        make_dataset.main()

    df = pd.read_parquet(config.INT_FILE_PATH / config.INT_FILE_NAME)

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

    logger.info("BINNING NUMERICAL DATA")

    for col in ["property_value", "ltv", "dtir1"]:
        df[f"{col}_binned"] = helpers.bin_values(df, col)
        dummies = pd.get_dummies(df[f"{col}_binned"], prefix=col, prefix_sep="_")
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=[col, f"{col}_binned", f"{col}_na"], inplace=True)

    logger.info("VECTORIZING CATEGORICAL DATA")

    for col in df.columns:
        if not types.is_categorical_dtype(df[col]):
            continue
        if col == config.TARGET:
            continue
        dummies = pd.get_dummies(df[col], prefix=col, prefix_sep="_", drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=col, inplace=True)

    logger.info("IMPUTING MISSING DATA")

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

    train, test = train_test_split(df, test_size=0.2, random_state=config.RANDOM_STATE)

    for df, name in zip([train, test], ["train", "test"]):
        df.to_parquet(config.FIN_FILE_PATH / f"{name}.parquet")

    logger.info(f"DONE EXPORTS")


if __name__ == "__main__":
    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
