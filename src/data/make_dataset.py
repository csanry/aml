import logging
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
from src import config, evaluation, helpers, plotting


def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger()
    logger.info("MAKING DATA SET FROM THE RAW DATA")

    raw_filename = "Loan_Default.csv"

    if os.path.exists(config.RAW_FILE_PATH / raw_filename):
        logger.info("READING FROM THE LOCAL COPY")
        df = pd.read_csv(config.RAW_FILE_PATH / raw_filename)
    else:
        logger.info(f"FILE DOES NOT EXIST: {raw_filename}")
        logger.info("DOWNLOADING DATA >>>")
        df = pd.read_csv("https://storage.googleapis.com/aml_1/Loan_Default.csv")
        df.to_csv(config.RAW_FILE_PATH / raw_filename)

    # standardize columns
    df.columns = helpers.standardize_cols(df.columns)

    # dropping columns that are not useful
    df = df.drop(
        columns=[
            "unnamed:_0",
            "id",
            "year",
            "interest_rate_spread",
            "secured_by",
            "construction_type",
            "security_type",
        ]
    )

    # convert to categorical
    cat_columns = [
        "gender",
        "loan_type",
        "loan_limit",
        "loan_purpose",
        "approv_in_adv",
        "neg_ammortization",
        "business_or_commercial",
        "credit_worthiness",
        "credit_type",
        "co_applicant_credit_type",
        "submission_of_application",
        "open_credit",
    ]
    for cat_col in cat_columns:
        df[cat_col] = helpers.convert_to_dtype(df[cat_col], "categorical")

    num_columns = ["property_value", "dtir1", "ltv", "loan_amount", "income"]
    for num_col in num_columns:
        df[num_col] = helpers.convert_to_dtype(df[num_col], type="numeric")

    # map lump_sum payment
    lump_sum_mapping = {"not_lpsm": False, "lpsm": True}
    df["lump_sum_payment"] = (
        df["lump_sum_payment"].map(lump_sum_mapping).astype("category")
    )

    # map interest_only
    interest_only_mapping = {"not_int": False, "int_only": True}
    df["interest_only"] = (
        df["interest_only"].map(interest_only_mapping).astype("category")
    )

    # map total_units
    total_units_cat = pd.CategoricalDtype(
        categories=["1U", "2U", "3U", "4U"], ordered=True
    )
    df["total_units"] = df["total_units"].astype(total_units_cat)

    # map occupancy_type
    occupancy_type_map = {
        "pr": "primary residence",
        "sr": "secondary residence",
        "ir": "investment residence",
    }
    df["occupancy_type"] = (
        df["occupancy_type"].map(occupancy_type_map).astype("category")
    )

    # convert region
    df["region"] = df["region"].str.lower().astype("category")

    # convert age
    age_bins = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]
    age_cat = pd.CategoricalDtype(categories=age_bins, ordered=True)
    df["age"] = df["age"].astype(age_cat)

    df.to_parquet(config.ROOT_DIR / "data" / "interim" / "df.parquet")


if __name__ == "__main__":
    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    main()
