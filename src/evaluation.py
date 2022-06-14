# evaluation.py
import os
from typing import Type, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
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
    TRAIN F_score: {tuner.cv_results_['mean_train_F_score'][tuner.best_index_]:.2%}
    TEST F_score: {tuner.cv_results_['mean_test_F_score'][tuner.best_index_]:.2%}  
    """
    )


def evaluate_report(
    y_test: pd.Series, y_pred: np.ndarray, y_pred_prob: np.ndarray
) -> dict:
    report = {}
    report["accuracy"] = accuracy_score(y_test, y_pred)
    report["precision"] = precision_score(y_test, y_pred)
    report["recall"] = recall_score(y_test, y_pred)
    report["f05"] = fbeta_score(y_test, y_pred, beta=0.5)
    report["f1"] = f1_score(y_test, y_pred)
    report["f2"] = fbeta_score(y_test, y_pred, beta=2)
    report["cf_matrix"] = confusion_matrix(y_test, y_pred, normalize="all")
    report["auroc"] = roc_auc_score(y_test, y_pred_prob)
    report["roc"] = roc_curve(y_test, y_pred_prob)

    print(
        f"""
    -----------
    PERFORMANCE 
    -----------
    ACCURACY: {report["accuracy"]:.2%}
    PRECISION: {report["precision"]:.2%}
    RECALL: {report["recall"]:.2%}
    F05: {report["f05"]:.2%}
    F1: {report["f1"]:.2%}
    F2: {report["f2"]:.2%}
    ROC AUC: {report["auroc"]:.2%}
    """
    )
    return report


def main() -> None:
    pass


if __name__ == "__main__":
    main()
