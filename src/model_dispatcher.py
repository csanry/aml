from src import config
from src.helpers import load_model

base_path = config.MODEL_OUTPUT_PATH

large_models = {
    "gbm_300000": load_model(base_path / "gbm_300000_large.pickle"),
    "log_reg_300000": load_model(base_path / "log_reg_300000_large.pickle"),
    "nca_300000": load_model(base_path / "nca_300000_large.pickle"),
    "rf_300000": load_model(base_path / "rf_300000_large.pickle"),
    "ada_300000": load_model(base_path / "adaboost_300000_large.pickle"),
    "svm_300000": load_model(base_path / "svm_300000_large.pickle"),
}

large_models_threshold_sel = {
    "gbm_200000": load_model(base_path / "gbm_200000_large.pickle"),
    "gbm_300000": load_model(base_path / "gbm_300000_large.pickle"),
    "gbm_400000": load_model(base_path / "gbm_400000_large.pickle"),
    "gbm_500000": load_model(base_path / "gbm_500000_large.pickle"),
    "rf_200000": load_model(base_path / "rf_200000_large.pickle"),
    "rf_300000": load_model(base_path / "rf_300000_large.pickle"),
    "rf_400000": load_model(base_path / "rf_400000_large.pickle"),
    "rf_500000": load_model(base_path / "rf_500000_large.pickle"),
}

small_models = {
    "gbm_300000": load_model(base_path / "gbm_300000_small.pickle"),
    "log_reg_300000": load_model(base_path / "log_reg_300000_small.pickle"),
    "nca_300000": load_model(base_path / "nca_300000_small.pickle"),
    "rf_300000": load_model(base_path / "rf_300000_small.pickle"),
    "ada_300000": load_model(base_path / "adaboost_300000_small.pickle"),
    "svm_300000": load_model(base_path / "svm_300000_small.pickle"),
    "gbm_200000": load_model(base_path / "gbm_200000_small.pickle"),
    "gbm_400000": load_model(base_path / "gbm_400000_small.pickle"),
    "gbm_500000": load_model(base_path / "gbm_500000_small.pickle"),
    "rf_200000": load_model(base_path / "rf_200000_small.pickle"),
    "rf_400000": load_model(base_path / "rf_400000_small.pickle"),
    "rf_500000": load_model(base_path / "rf_500000_small.pickle"),
}

small_models_threshold_sel = {
    "gbm_200000": load_model(base_path / "gbm_200000_small.pickle"),
    "gbm_300000": load_model(base_path / "gbm_300000_small.pickle"),
    "gbm_400000": load_model(base_path / "gbm_400000_small.pickle"),
    "gbm_500000": load_model(base_path / "gbm_500000_small.pickle"),
    "rf_200000": load_model(base_path / "rf_200000_small.pickle"),
    "rf_300000": load_model(base_path / "rf_300000_small.pickle"),
    "rf_400000": load_model(base_path / "rf_400000_small.pickle"),
    "rf_500000": load_model(base_path / "rf_500000_small.pickle"),
}
