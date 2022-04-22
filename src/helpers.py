# helpers.py 
import os 
import sys
import pandas as pd 
import pandas.api.types as types
from typing import List, Set, Dict, Tuple
from typing import Union, Any, Optional, Iterable, Hashable
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import missingno as msno


def get_numeric_columns(df: pd.DataFrame) -> List: 
    """

    Parameters
    ----------
    df : pd.DataFrame :
        
    df : pd.DataFrame :
        
    df : pd.DataFrame :
        
    df: pd.DataFrame :
        

    Returns
    -------

    
    """
    return [col for col in df.columns if types.is_numeric_dtype(df[col])]


def get_categorical_columns(df: pd.DataFrame) -> List: 
    """

    Parameters
    ----------
    df : pd.DataFrame :
        
    df : pd.DataFrame :
        
    df : pd.DataFrame :
        
    df: pd.DataFrame :
        

    Returns
    -------

    
    """
    return [col for col in df.columns if types.is_categorical_dtype(df[col])]


def convert_to_dtype(col: pd.Series, type: str = 'categorical') -> pd.Series:
    """

    Parameters
    ----------
    col : pd.Series :
        
    type : str :
        (Default value = 'categorical')
    col : pd.Series :
        
    type : str :
        (Default value = 'categorical')
    col : pd.Series :
        
    type : str :
        (Default value = 'categorical')
    col: pd.Series :
        
    type: str :
         (Default value = 'categorical')

    Returns
    -------

    
    """
    if type not in ['numeric', 'categorical']: 
        raise ValueError('Please enter a valid dtype of either: "numeric" or "categorical"') 
    elif type == 'numeric': 
        return pd.to_numeric(col, errors='raise')
    return col.astype('category')


def replace_missing_values(df: pd.DataFrame, cols: Union[str, Iterable[str], Hashable], value) -> pd.DataFrame:
    """

    Parameters
    ----------
    df : pd.DataFrame :
        
    cols : Union[str :
        
    Iterable[str] :
        
    Hashable] :
        
    value :
        
    df : pd.DataFrame :
        
    cols : Union[str :
        
    df : pd.DataFrame :
        
    cols : Union[str :
        
    df: pd.DataFrame :
        
    cols: Union[str :
        

    Returns
    -------

    
    """
    return df.fillna(value={cols: value})

def return_value_counts(df: pd.DataFrame) -> None: 
    """

    Parameters
    ----------
    df : pd.DataFrame :
        
    df : pd.DataFrame :
        
    df : pd.DataFrame :
        
    df: pd.DataFrame :
        

    Returns
    -------

    
    """

    for col in df.columns: 
        print(col.upper())
        print('####################################')
        print(df[col].value_counts())
        print('\n')

def set_up_fig(rows: int=1) -> None: 
    """

    Parameters
    ----------
    rows : int :
        (Default value = 1)
    rows : int :
        (Default value = 1)
    rows : int :
        (Default value = 1)
    rows: int :
         (Default value = 1)

    Returns
    -------

    
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,9))
    for s in ['top', 'right']: 
        ax.spines[s].set_visible(False)

def quick_eda(df: pd.DataFrame) -> None: 
    """

    Parameters
    ----------
    df : pd.DataFrame :
        
    df : pd.DataFrame :
        
    df : pd.DataFrame :
        
    df: pd.DataFrame :
        

    Returns
    -------

    
    """
    print(f'DATAFRAME HAS {df.shape[0]} ROWS AND {df.shape[1]} COLS')
    print(df.info())
    display(df.describe().T)
    display(df.head(5))


def standardize_cols(column_list: List[str]) -> List[str]: 
    """

    Parameters
    ----------
    column_list: List[str] :
        

    Returns
    -------
    """
    return [col.lower().strip().replace(' ', '-') for col in column_list]

def visualize_cols(column_list: List[str]) -> List[str]: 
    """

    Parameters
    ----------
    column_list: List[str] :
        

    Returns
    -------

    """

    return [col.capitalize().replace('-', ' ') for col in column_list]


def missingness_checks(df: pd.DataFrame) -> None:
    """

    Parameters
    ----------
    df : pd.DataFrame :
        
    df : pd.DataFrame :
        
    df: pd.DataFrame :
        

    Returns
    -------

    
    """
    
    print(f'NUMBER OF MISSING COLUMNS: {df.isna().sum().sum()}')
    print(f'MISSING COLUMNS (0: NO MISSING VALUES, 1: MISSING VALUES')
    print(df.isna().sum())
    print('\n')
    print('MISSINGNESS THROUGHOUT THE DATA')
    msno.matrix(df) 
    plt.show()
    print('MISSINGNESS CORRELATIONS')
    msno.heatmap(df)
    plt.show() 

def get_dtypes(df: pd.DataFrame) -> Dict:
    """

    Parameters
    ----------
    df : pd.DataFrame :
        
    df : pd.DataFrame :
        
    df: pd.DataFrame :
        

    Returns
    -------

    
    """
    return df.dtypes.to_dict()

def quick_plot(df: pd.DataFrame, hue_var: str = None, diag_kind: str = 'kde') -> None: 
    """

    Parameters
    ----------
    df : pd.DataFrame :
        
    hue_var : str :
        (Default value = None)
    diag_kind : str :
        (Default value = 'kde')
    df : pd.DataFrame :
        
    hue_var : str :
        (Default value = None)
    diag_kind : str :
        (Default value = 'kde')
    df: pd.DataFrame :
        
    hue_var: str :
         (Default value = None)
    diag_kind: str :
         (Default value = 'kde')

    Returns
    -------

    
    """
    sns.pairplot(df, hue=hue_var, diag_kind=diag_kind)
    plt.show() 

def main() -> None: 
    """ """
    pass 

if __name__ == "__main__": 
    main() 