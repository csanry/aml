import logging
import pickle
import warnings

import numpy as np
from sklearn import metrics
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
        refit="F2",
        cv=cv_split,
        return_train_score=True,
        n_jobs=config.N_JOBS,
        verbose=10,
    )

    knn_cv.fit(X_train, y_train)

    knn_best_pipe = knn_cv.best_estimator_

    return knn_cv, knn_best_pipe


def evaluate(X_test, y_test, knn_cv, knn_best_pipe):

    evaluation.evaluate_tuning(tuner=knn_cv)
    knn_y_pred_prob = knn_best_pipe.predict_proba(X_test)[:, 1]
    knn_y_pred = knn_best_pipe.predict(X_test)

    evaluation.evaluate_report(
        y_test=y_test, y_pred=knn_y_pred, y_pred_prob=knn_y_pred_prob
    )

    cf_matrix = metrics.confusion_matrix(y_test, knn_y_pred)
    plotting.plot_confusion_matrix(cf_matrix, "knn")

    filename = config.MODEL_OUTPUT_PATH / "knn.pickle"
    with open(filename, "wb") as file:
        pickle.dump(knn_best_pipe, file)

