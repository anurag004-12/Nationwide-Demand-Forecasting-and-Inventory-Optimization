import lightgbm as lgb
from src.config import QUANTILES
from src.features import FEATURES, CATEGORICAL


def train_point_model(train):
    """Main LightGBM model. Tweedie loss suits counts with many zeros."""
    params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.1,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
    }
    data = lgb.Dataset(train[FEATURES], train["sales"], categorical_feature=CATEGORICAL)
    return lgb.train(params, data, num_boost_round=500)


def train_quantile_models(train):
    """One model per quantile (5%, 25%, 50%, 75%, 95%) to get prediction intervals."""
    models = {}
    for q in QUANTILES:
        params = {
            "objective": "quantile",
            "alpha": q,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "verbose": -1,
        }
        data = lgb.Dataset(train[FEATURES], train["sales"], categorical_feature=CATEGORICAL)
        models[q] = lgb.train(params, data, num_boost_round=300)
    return models