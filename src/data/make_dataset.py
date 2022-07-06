import logging
import os
import warnings

import pandas as pd
from src import config, helpers

warnings.filterwarnings("ignore")


def make_dataset(
    data_source: str = "https://storage.googleapis.com/aml_0/Loan_Default.csv",
) -> None:
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger()
    logger.info("MAKING DATA SET FROM THE RAW DATA")

    if os.path.exists(config.RAW_FILE_PATH / config.RAW_FILE_NAME):
        logger.info("READING FROM THE LOCAL COPY")
        df = pd.read_csv(config.RAW_FILE_PATH / config.RAW_FILE_NAME, index_col=0)
    else:
        logger.info(f"FILE DOES NOT EXIST: {config.RAW_FILE_NAME}")
        logger.info("DOWNLOADING DATA ")
        df = pd.read_csv(data_source, index_col=0)
        df.to_csv(config.RAW_FILE_PATH / config.RAW_FILE_NAME)
        logger.info("RAW FILE PLACED IN RAW FOLDER AND READY FOR BASIC TRANSFORMATIONS")

    # standardize columns
    df.columns = helpers.standardize_cols(df.columns)

    # convert to numeric
    num_columns = [
        "id",
        "interest_rate_spread",
        "property_value",
        "dtir1",
        "ltv",
        "loan_amount",
        "income",
    ]

    for num_col in num_columns:
        df[num_col] = helpers.convert_to_dtype(df[num_col], type="numeric")

    # map columns
    gender_mapping = {
        "Male": "m",
        "Female": "f",
        "Joint": "j",
        "Sex Not Available": "na",
    }
    loan_type_mapping = {"type1": "t1", "type2": "t2", "type3": "t3"}
    loan_limit_mapping = {"ncf": 0, "cf": 1}
    approv_in_adv_mapping = {"nopre": 0, "pre": 1}
    neg_ammortization_mapping = {"not_neg": 0, "neg_amm": 1}
    business_or_commercial_mapping = {"b/c": 1, "nob/c": 0}
    lump_sum_payment_mapping = {"not_lpsm": 0, "lpsm": 1}
    interest_only_mapping = {"not_int": 0, "int_only": 1}
    credit_worthiness_mapping = {"l1": 1, "l2": 0}
    submission_of_application_mapping = {"to_inst": 1, "not_inst": 0}
    open_credit_mapping = {"opc": 1, "nopc": 0}
    total_units_mapping = {"1U": 1, "2U": 2, "3U": 3, "4U": 4}

    cols_to_map = [
        "gender",
        "loan_type",
        "loan_limit",
        "approv_in_adv",
        "neg_ammortization",
        "business_or_commercial",
        "lump_sum_payment",
        "interest_only",
        "credit_worthiness",
        "submission_of_application",
        "open_credit",
        "total_units",
    ]

    mappings = [
        gender_mapping,
        loan_type_mapping,
        loan_limit_mapping,
        approv_in_adv_mapping,
        neg_ammortization_mapping,
        business_or_commercial_mapping,
        lump_sum_payment_mapping,
        interest_only_mapping,
        credit_worthiness_mapping,
        submission_of_application_mapping,
        open_credit_mapping,
        total_units_mapping,
    ]

    for col, mapping in zip(cols_to_map, mappings):
        df[col] = df[col].map(mapping)

    # convert to categorical
    cat_columns = [
        "year",
        "gender",
        "loan_type",
        "loan_limit",
        "loan_purpose",
        "approv_in_adv",
        "neg_ammortization",
        "business_or_commercial",
        "credit_worthiness",
        "credit_type",
        "secured_by",
        "construction_type",
        "security_type",
        "co_applicant_credit_type",
        "submission_of_application",
        "lump_sum_payment",
        "interest_only",
        "occupancy_type",
        "open_credit",
    ]

    for cat_col in cat_columns:
        df[cat_col] = helpers.convert_to_dtype(df[cat_col], type="categorical")

    # processing occupancy
    occupancy_type_map = {
        "pr": "primary",
        "sr": "secondary",
        "ir": "investment",
    }
    df["occupancy_type"] = df["occupancy_type"].map(occupancy_type_map)

    # processing region
    df["region"] = df["region"].str.lower().astype("category")

    # processing age
    age_bins = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]
    age_cat = pd.CategoricalDtype(categories=age_bins, ordered=True)
    df["age"] = df["age"].astype(age_cat)

    # extract file to INT
    df.to_parquet(config.INT_FILE_PATH / config.BASIC_CLEAN_FILE_NAME)
    logger.info("CLEAN FILE PLACED IN INTERIM AND READY FOR THRESHOLD SPLIT")


if __name__ == "__main__":

    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    make_dataset(data_source="https://storage.googleapis.com/aml_0/Loan_Default.csv")

