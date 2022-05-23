import logging
import pickle
import warnings

from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from src import config, evaluation, helpers

warnings.filterwarnings("ignore")


def main():

    logger = logging.getLogger()

    scorer = config.SCORER
    cv_split = config.CV_SPLIT

    X_train, X_test, y_train, y_test = helpers.read_files()

    logger.info("HYPERPARAMETER TUNING")

    # Setup the hyperparameter grid
    svm_param_grid = {
        "pca__n_components": [10, 15],
        "svm__C": [0.01, 0.1, 1, 10, 100],
        "svm__kernel": ["rbf", "linear",],
    }

    # baseline model
    svm = SVC(
        probability=True, max_iter=5000, random_state=config.RANDOM_STATE, verbose=True
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
        refit="F2",
        cv=cv_split,
        return_train_score=True,
        n_jobs=config.N_JOBS,
        verbose=10,
        random_state=config.RANDOM_STATE,
    )

    svm_cv.fit(X_train, y_train)

    evaluation.evaluate_tuning(tuner=svm_cv)

    n_components, C, kernel = (
        svm_cv.best_params_.get("pca__n_components"),
        svm_cv.best_params_.get("svm__C"),
        svm_cv.best_params_.get("svm__kernel"),
    )

    svm_best = SVC(
        C=C,
        kernel=kernel,
        probability=True,
        max_iter=5000,
        random_state=config.RANDOM_STATE,
        verbose=True,
    )

    pca_best = PCA(n_components=n_components, random_state=config.RANDOM_STATE,)

    # build the best pipeline
    svm_best_pipe = Pipeline(
        [("mm", mm_scale), ("pca_best", pca_best), ("svm_best", svm_best)]
    )

    svm_best_pipe.fit(X_train, y_train)

    svm_y_pred_prob = svm_best_pipe.predict_proba(X_test)[:, 1]
    svm_y_pred = svm_best_pipe.predict(X_test)

    evaluation.evaluate_report(
        y_test=y_test, y_pred=svm_y_pred, y_pred_prob=svm_y_pred_prob
    )

    filename = config.MODEL_OUTPUT_PATH / "svm.pickle"
    with open(filename, "wb") as file:
        pickle.dump(svm_best_pipe, file)

    logger.info("DONE")


if __name__ == "__main__":
    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
