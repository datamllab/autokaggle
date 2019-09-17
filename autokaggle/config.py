from sklearn.base import BaseEstimator
from autokaggle.utils import rand_temp_folder_generator, ensure_dir
from hyperopt import hp
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor,\
    ExtraTreesRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMClassifier, LGBMRegressor


class Config(BaseEstimator):
    def __init__(self, path=None, verbose=True, time_limit=None, use_ensembling=True, num_estimators_ensemble=50,
                 ensemble_strategy='ranked_ensembling', ensemble_method='max_voting', search_iter=500, cv_folds=3,
                 subsample_ratio=0.1, random_ensemble=False):
        self.verbose = verbose
        self.path = path if path is not None else rand_temp_folder_generator()
        ensure_dir(self.path)
        if self.verbose:
            print('Path:', self.path)
        self.time_limit = time_limit
        self.objective = None
        self.use_ensembling = use_ensembling
        self.hparams = None
        self.num_estimators_ensemble = num_estimators_ensemble
        self.ensemble_strategy = ensemble_strategy
        self.ensemble_method = ensemble_method
        self.random_ensemble = random_ensemble
        self.search_iter = search_iter
        self.cv_folds = cv_folds
        self.subsample_ratio = subsample_ratio
        self.resampling_strategy = 'auto'
        self.random_state = 1001
        self.classification_models = ['knn', 'svm', 'lgbm', 'random_forest', 'adaboost']
        self.regression_models = ['extratree', 'ridge', 'lgbm', 'random_forest', 'adaboost']

    def update(self, options):
        for k, v in options.items():
            if hasattr(self, k):
                setattr(self, k, v)


knn_classifier_params = {
    'n_neighbors': hp.choice('n_neighbors', range(2, 20)),
    'algorithm': hp.choice('algorithm', ['ball_tree', 'kd_tree', 'brute']),
    'leaf_size': hp.choice('leaf_size', range(5, 50)),
    'metric': hp.choice('metric', ["euclidean", "manhattan", "chebyshev", "minkowski"]),
    'p': hp.choice('p', range(1, 4)),
}

svc_params = {
    'C': hp.lognormal('C', 0, 1),
    'kernel': hp.choice('kernel', ['rbf', 'poly', 'linear', 'sigmoid']),
    'degree': hp.choice('degree', range(1, 6)),
    'gamma': hp.uniform('gamma', 0.001, 10000),
    'max_iter': 50000,
}

random_forest_classifier_params = {
    'criterion': hp.choice('criterion', ['entropy', 'gini']),
    'max_features': hp.uniform('max_features', 0, 1.0),
    'n_estimators': hp.choice('rf_n_estimators', range(50, 200)),
    'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 10))
}

lgbm_classifier_params = {
    'boosting_type': 'gbdt',
    'min_split_gain': 0.1,
    'subsample': 0.8,
    'num_leaves': 80,
    'colsample_bytree': hp.uniform('colsample_bytree', 0.4, 0.8),
    'min_child_weight': hp.choice('min_child_weight', range(1, 100)),
    'max_depth': hp.choice('max_depth', range(5, 10)),
    'n_estimators': hp.choice('n_estimators', range(50, 200)),
    'learning_rate': hp.lognormal('learning_rate', 0, 1),
}

adaboost_classifier_params = {
    'algorithm': hp.choice('algorithm_adaboost', ['SAMME.R', 'SAMME']),
    'n_estimators': hp.choice('n_estimators_adaboost', range(50, 200)),
    'learning_rate': hp.lognormal('learning_rate_adaboost', 0, 1),
}

extra_trees_regressor_params = {
    'n_estimators': hp.choice('n_estimators_extra_trees', range(50, 200)),
    'criterion': hp.choice('criterion_extra_trees', ['mse', 'friedman_mse', 'mae']),
    'max_features': hp.uniform('max_features_extra_trees', 0, 1.0),
    'min_samples_leaf': hp.choice('min_samples_leaf_extra_trees', range(1, 10)),
    'min_impurity_decrease': 0.0
}

ridge_params = {
    'fit_intercept': True,
    'tol': hp.loguniform('tol_ridge', 1e-5, 1e-1),
    'alpha': hp.loguniform('alpha_ridge', 1e-5, 10)
}

random_forest_regressor_params = {
    'criterion': hp.choice('criterion', ['mse', 'friedman_mse', 'mae']),
    'max_features': hp.uniform('max_features', 0, 1.0),
    'n_estimators': hp.choice('rf_n_estimators', range(50, 200)),
    'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 10))
}

lgbm_regressor_params = {
    'boosting_type': 'gbdt',
    'min_split_gain': 0.1,
    'subsample': 0.8,
    'num_leaves': 80,
    'colsample_bytree': hp.uniform('colsample_bytree', 0.4, 0.8),
    'min_child_weight': hp.choice('min_child_weight', range(1, 100)),
    'max_depth': hp.choice('max_depth', range(5, 10)),
    'n_estimators': hp.choice('n_estimators', range(50, 200)),
    'learning_rate': hp.lognormal('learning_rate', 0, 1),
}

adaboost_regressor_params = {
    'loss': hp.choice('loss_adaboost', ["linear", "square", "exponential"]),
    'n_estimators': hp.choice('n_estimators_adaboost', range(50, 200)),
    'learning_rate': hp.lognormal('learning_rate_adaboost', 0, 1),
}


regression_hspace = {
    'extratree': {
        'model': ExtraTreesRegressor,
        'param': extra_trees_regressor_params
    },
    'ridge': {
        'model': Ridge,
        'param': ridge_params
    },
    'random_forest': {
        'model': RandomForestRegressor,
        'param': random_forest_regressor_params
    },
    'lgbm': {
        'model': LGBMRegressor,
        'param': lgbm_regressor_params
    },
    'adaboost': {
        'model': AdaBoostRegressor,
        'param': adaboost_regressor_params
     }
}


classification_hspace = {
    'knn': {
        'model': KNeighborsClassifier,
        'param': knn_classifier_params
    },
    'svm': {
        'model': SVC,
        'param': svc_params
    },
    'random_forest': {
        'model': RandomForestClassifier,
        'param': random_forest_classifier_params
    },
    'lgbm': {
        'model': LGBMClassifier,
        'param': lgbm_classifier_params
    },
    'adaboost': {
        'model': AdaBoostClassifier,
        'param': adaboost_classifier_params
    }
}