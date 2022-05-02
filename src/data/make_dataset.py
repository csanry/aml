import click
import logging
import os
from pathlib import Path
import numpy as np 
import pandas as pd

from src import helpers, config, plotting, evaluation

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger()
    logger.info('MAKING DATA SET FROM THE RAW DATA')
    
    raw_filename = 'Loan_Default.csv'

    if os.path.exists(config.RAW_FILE_PATH / raw_filename): 
        logger.info('READING FROM THE LOCAL COPY')
        df = pd.read_csv(config.RAW_FILE_PATH / raw_filename)
    else: 
        logger.info(f'FILE DOES NOT EXIST: {raw_filename}')
        logger.info('DOWNLOADING DATA >>>')
        df = pd.read_csv('https://storage.googleapis.com/aml_1/Loan_Default.csv')
        df.to_csv(config.RAW_FILE_PATH / raw_filename)
   

    df.to_parquet(config.ROOT_DIR / "data" / "interim" / "df.parquet")  


if __name__ == '__main__':
    log_fmt = '%(asctime)s:%(name)s:%(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    main()