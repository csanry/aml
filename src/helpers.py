# helpers.py
import os
import pickle
import sys
from ast import Index
from typing import (Any, Dict, Hashable, Iterable, List, Optional, Set, Tuple,
                    Union)

import matplotlib as mpl
import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
import pandas.api.types as types
import seaborn as sns

from src import config


def quick_eda(df: pd.DataFrame) -> None:
    """Prints out quick summary statistics and information about the data

    Parameters
    ----------
    df : pd.DataFrame :
        pandas DataFrame object

    Returns
    -------
    None; prints information
    """
    print(
        f"""
    DATAFRAME HAS {df.shape[0]} ROWS AND {df.shape[1]} COLS
    {df.info()}
    """
    )
    display(df.describe().T)
    display(df.head(5))


def get_categorical_columns(df: pd.DataFrame) -> List:
    """Gets a list of columns in the DataFrame which are categorical
    
    Parameters
    ----------
    df : pd.DataFrame :
        pandas DataFrame object
        
    Returns
    -------
    A list of column names which are of type categorical
    
    """
    return [col for col in df.columns if types.is_categorical_dtype(df[col])]


def get_numeric_columns(df: pd.DataFrame) -> List:
    """Gets a list of columns in the DataFrame which are numeric 

    Parameters
    ----------
    df : pd.DataFrame :
        pandas DataFrame object

    Returns
    -------
    A list of column names which are of type numeric
    """
    return [col for col in df.columns if types.is_numeric_dtype(df[col])]


def bin_values(df: pd.DataFrame, col: str) -> pd.Series:
    """Bins values of a numeric column into 5 quantiles, then fills in missing 

    Parameters
    ----------
    df : pd.DataFrame :
        pandas DataFrame object

    col : str : 
        column to bin values on 

    Returns
    -------
    None
    """
    column_binned = pd.qcut(df[col], q=5, labels=["1", "2", "3", "4", "5"])
    column_binned = column_binned.astype("object")
    column_binned = column_binned.fillna(value="na")
    return column_binned


def read_files():
    train = pd.read_parquet(config.FIN_FILE_PATH / "train.parquet")
    test = pd.read_parquet(config.FIN_FILE_PATH / "test.parquet")
    X_train, y_train = train.drop(columns=config.TARGET), train[config.TARGET]
    X_test, y_test = test.drop(columns=config.TARGET), test[config.TARGET]
    return (X_train, X_test, y_train, y_test)


def convert_to_dtype(col: pd.Series, type: str = "categorical") -> pd.Series:
    """Convert a column to a dtype 

    Parameters
    ----------
    type : str :
        (Default value = 'categorical')
        Specifies the type of conversion to make - either 'categorical' or 'numeric' 
    col : pd.Series :
        Input column 
    Returns
    -------
    A list of column of type that is converted to 
    
    """
    if type not in ["numeric", "categorical"]:
        raise ValueError(
            'Please enter a valid dtype of either: "numeric" or "categorical"'
        )
    elif type == "numeric":
        return pd.to_numeric(col, errors="raise")
    return col.astype("category")


def replace_missing_values(
    df: pd.DataFrame, cols: Union[str, Iterable[str], Hashable], value
) -> pd.DataFrame:
    """Replaces missing values 

    Parameters
    ----------
    df : pd.DataFrame :
        
    cols : Union[str :
        
    Iterable[str] :
        
    Hashable] :
        
    value :
        
    Returns
    -------

    """
    return df.fillna(value={cols: value})


def return_value_counts(df: pd.DataFrame) -> None:
    """Compute value counts for each column and prints it out

    Parameters
    ----------
    df : pd.DataFrame :
        pandas DataFrame object

    Returns
    -------
    None; prints value counts for each column
    
    """
    for col in df.columns:
        print(
            f"""
        {col.upper()}
        ####################################
        {df[col].value_counts()}

        """
        )


def standardize_cols(column_list: Index) -> List[str]:
    """Transforms column names into a standardized format for data analysis

    Parameters
    ----------
    column_list: List[str] :
        List of column names

    Returns
    -------
    List of transformed column names
    """
    return [
        col.lower().strip().replace(" ", "_").replace("-", "_") for col in column_list
    ]


def visualize_cols(column_list: List[str]) -> List[str]:
    """Transforms column names into a standardized format for data visualization

    Parameters
    ----------
    column_list: List[str] :
        List of column names

    Returns
    -------
    List of transformed column names
    """
    return [col.capitalize().replace("_", " ") for col in column_list]


def missingness_checks(df: pd.DataFrame) -> None:
    """Perform missingness checks on the data

    Parameters
    ----------
    df : pd.DataFrame :
        pandas DataFrame object
    
    Returns
    -------
    Two plot outputs and missingness information
    """

    print(
        f"""
    NUMBER OF MISSING COLUMNS: {df.isna().sum().sum()}
    MISSING COLUMNS (0: NO MISSING VALUES, 1: MISSING VALUES) 
    {df.isna().sum()}

    MISSINGNESS THROUGHOUT THE DATA
    ####################################
    """
    )
    msno.matrix(df)
    plt.show()
    print("MISSINGNESS CORRELATIONS")
    msno.heatmap(df)
    plt.show()


def get_dtypes(df: pd.DataFrame) -> Dict:
    """Get the dtypes of each column in a dictionary format

    Parameters
    ----------
    df : pd.DataFrame :
        pandas DataFrame object

    Returns
    -------
    Dictionary of column-dtype key-value pairs
    """
    return df.dtypes.to_dict()


def load_model(path_to_file):
    with open(path_to_file, "rb") as file:
        return pickle.load(file)


def main() -> None:
    """ """
    pass


if __name__ == "__main__":
    main()

