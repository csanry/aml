# evaluation.py
import os
from typing import Type, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, fbeta_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def evaluate_tuning(tuner: Union[Type[RandomizedSearchCV], Type[GridSearchCV]]) -> None:
    print(
        f"""
    --------------
    TUNING RESULTS
    --------------
    ESTIMATOR: {tuner.estimator}
    BEST SCORE: {tuner.best_score_:.2%}
    BEST PARAMS: {tuner.best_params_}
    TRAIN AUC: {tuner.cv_results_["mean_train_AUC"][tuner.best_index_]:.2%}
    TRAIN AUC SD: {tuner.cv_results_["std_train_AUC"][tuner.best_index_]:.2%}
    TEST AUC: {tuner.cv_results_["mean_test_AUC"][tuner.best_index_]:.2%}
    TEST AUC SD: {tuner.cv_results_["std_test_AUC"][tuner.best_index_]:.2%}
    TRAIN F2: {tuner.cv_results_['mean_train_F2'][tuner.best_index_]:.2%}
    TEST F2: {tuner.cv_results_['mean_test_F2'][tuner.best_index_]:.2%}  
    """
    )


def evaluate_report(
    y_test: pd.Series, y_pred: np.ndarray, y_pred_prob: np.ndarray
) -> None:
    print(
        f"""
    -----------
    PERFORMANCE 
    -----------
    ACCURACY: {accuracy_score(y_test, y_pred):.2%}
    PRECISION: {precision_score(y_test, y_pred):.2%}
    RECALL: {recall_score(y_test, y_pred):.2%}
    F1: {f1_score(y_test, y_pred):.2%}
    F2: {fbeta_score(y_test, y_pred, beta=2):.2%}
    ROC AUC: {roc_auc_score(y_test, y_pred_prob):.2%}
    """
    )


def main() -> None:
    pass


if __name__ == "__main__":
    main()
