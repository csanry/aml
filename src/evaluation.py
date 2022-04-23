# evaluation.py
import os
import numpy as np
import pandas as pd 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from typing import Union, Type


def evaluate_tuning_results(tuner: Union[Type[GridSearchCV], Type[RandomizedSearchCV]]) -> None: 
    print('TUNING RESULTS')
    print('####################################')
    print(f'ESTIMATOR: {tuner.estimator}')
    print(f'BEST SCORE: {tuner.best_score_:.2%}')
    print(f'BEST PARAMS: {tuner.best_params_}')
    print(f'AVERAGE TRAIN SCORE: {tuner.cv_results_["mean_train_score"].mean():.2%}')
    print(f'AVERAGE TRAIN SD: {tuner.cv_results_["std_train_score"].mean():.2%}')
    print(f'AVERAGE TEST SCORE: {tuner.cv_results_["mean_test_score"].mean():.2%}')
    print(f'AVERAGE TEST SD: {tuner.cv_results_["std_test_score"].mean():.2%}')

def main() -> None: 
    pass

if __name__ == '__main__': 
    main() 