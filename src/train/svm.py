import logging
import pickle
import warnings

from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from src import config, evaluation, plotting

warnings.filterwarnings("ignore")


def train(X_train, y_train, scorer, cv_split):

    # Setup the hyperparameter grid
    svm_param_grid = {
        "pca__n_components": [10, 15],
        "svm__C": [0.01, 0.1, 1, 10, 100],
        "svm__kernel": ["rbf", "linear",],
    }

    # baseline model
    svm = SVC(
        probability=True, max_iter=3000, random_state=config.RANDOM_STATE, verbose=True
    )
    mm_scale = MinMaxScaler(feature_range=(0, 1))
    pca = PCA(random_state=config.RANDOM_STATE,)

    # build the pipeline
    svm_pipe = Pipeline([("mm", mm_scale), ("pca", pca), ("svm", svm)])

    # Cross validate model with GridSearch
    svm_cv = RandomizedSearchCV(
        estimator=svm_pipe,
        param_distributions=svm_param_grid,
        n_iter=15,
        scoring=scorer,
        refit="F_score",
        cv=cv_split,
        return_train_score=True,
        n_jobs=config.N_JOBS,
        verbose=10,
        random_state=config.RANDOM_STATE,
    )

    svm_cv.fit(X_train, y_train)

    svm_best_pipe = svm_cv.best_estimator_

    return svm_cv, svm_best_pipe


def evaluate(svm_cv, svm_best_pipe, X_test, y_test, file_name):

    evaluation.evaluate_tuning(tuner=svm_cv)
    svm_y_pred_prob = svm_best_pipe.predict_proba(X_test)[:, 1]
    svm_y_pred = svm_best_pipe.predict(X_test)

    report = evaluation.evaluate_report(
        y_test=y_test, y_pred=svm_y_pred, y_pred_prob=svm_y_pred_prob
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
        pickle.dump(svm_best_pipe, file)

