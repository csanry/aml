import logging
import pickle
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from src import config, evaluation, plotting

warnings.filterwarnings("ignore")


def train(X_train, y_train, scorer, cv_split):
    # Setup the hyperparameter grid
    knn_param_grid = {
        "pca__n_components": [10, 15],
        "knn__n_neighbors": np.arange(3, 8),
    }

    # baseline model
    knn = KNeighborsClassifier(n_jobs=config.N_JOBS,)
    mm_scale = MinMaxScaler(feature_range=(0, 1))
    pca = PCA(random_state=config.RANDOM_STATE,)

    # build the pipeline
    knn_pipe = Pipeline([("mm", mm_scale), ("pca", pca), ("knn", knn)])

    # Cross validate model with GridSearch
    knn_cv = GridSearchCV(
        estimator=knn_pipe,
        param_grid=knn_param_grid,
        scoring=scorer,
        refit="F_score",
        cv=cv_split,
        return_train_score=True,
        n_jobs=config.N_JOBS,
        verbose=10,
    )

    knn_cv.fit(X_train, y_train)

    knn_best_pipe = knn_cv.best_estimator_

    return knn_cv, knn_best_pipe


def evaluate(knn_cv, knn_best_pipe, X_test, y_test, file_name):

    evaluation.evaluate_tuning(tuner=knn_cv)
    knn_y_pred_prob = knn_best_pipe.predict_proba(X_test)[:, 1]
    knn_y_pred = knn_best_pipe.predict(X_test)

    report = evaluation.evaluate_report(
        y_test=y_test, y_pred=knn_y_pred, y_pred_prob=knn_y_pred_prob
    )

    plotting.plot_confusion_matrix(cf_matrix=report["cf_matrix"], model_name=file_name)
    plotting.plot_roc_curve(
        fpr=report["roc"][0],
        tpr=report["roc"][1],
        model_name=file_name,
        auc=report["auroc"],
    )

    save_path = config.MODEL_OUTPUT_PATH / f"{file_name}.pickle"

    with open(save_path, "wb") as file:
        pickle.dump(knn_best_pipe, file)

