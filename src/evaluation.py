# evaluation.py
import os
import numpy as np
import pandas as pd 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from typing import Union, Type


def evaluate_tuning_results(tuner: Union[Type[RandomizedSearchCV], Type[GridSearchCV]]) -> None: 
    print(
    f'''
    TUNING RESULTS
    ####################################
    ESTIMATOR: {tuner.estimator}
    BEST SCORE: {tuner.best_score_:.2%}
    BEST PARAMS: {tuner.best_params_}
    AVERAGE TRAIN SCORE: {tuner.cv_results_["mean_train_score"].mean():.2%}
    AVERAGE TRAIN SD: {tuner.cv_results_["std_train_score"].mean():.2%}
    AVERAGE TEST SCORE: {tuner.cv_results_["mean_test_score"].mean():.2%}
    AVERAGE TEST SD: {tuner.cv_results_["std_test_score"].mean():.2%}
    ''')


def evaluate_report(y_test: pd.Series, y_pred: np.ndarray) -> None: 
    print(
    f'''
    PERFORMANCE 
    ####################################
    ACCURACY: {accuracy_score(y_test, y_pred):.2%}
    PRECISION: {precision_score(y_test, y_pred):.2%}
    RECALL: {recall_score(y_test, y_pred):.2%}
    F1: {f1_score(y_test, y_pred):.2%}
    ROC AUC: {roc_auc_score(y_test, y_pred):.2%}
    ''')

def main() -> None: 
    pass

if __name__ == '__main__': 
    main() 