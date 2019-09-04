import numpy as np
from pandas import DataFrame
import scipy
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PowerTransformer, KBinsDiscretizer
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from abc import abstractmethod
import collections
from lightgbm import LGBMClassifier, LGBMRegressor
LEVEL_HIGH = 32


class TabularPreprocessor:
    def __init__(self):
        """
        Initialization function for tabular preprocessor.
        """
        self.num_cat_pair = {}

        self.total_samples = 0

        self.cat_to_int_label = {}
        self.n_first_batch_keys = {}
        self.high_level_cat_keys = []

        self.feature_add_high_cat = 0
        self.feature_add_cat_num = 10
        self.feature_add_cat_cat = 10
        self.order_num_cat_pair = {}

        self.selected_cols = None
        self.budget = None
        self.data_info = None
        self.n_time = None
        self.n_num = None
        self.n_cat = None
        self.cat_col = None
        self.num_col = None
        self.time_col = None
        self.pipeline = None

    def fit(self, raw_x, y, time_limit, data_info):
        """
        This function should train the model parameters.

        Args:
            raw_x: a numpy.ndarray instance containing the training data.
            y: training label vector.
            time_limit: remaining time budget.
            data_info: meta-features of the dataset, which is an numpy.ndarray describing the
             feature type of each column in raw_x. The feature type include:
                     'TIME' for temporal feature, 'NUM' for other numerical feature,
                     and 'CAT' for categorical feature.
        """
        self.budget = time_limit
        # Extract or read data info
        self.data_info = data_info if data_info is not None else self.extract_data_info(raw_x)
        print('DATA_INFO: {}'.format(self.data_info))

        # Set the meta info for each data type
        self.n_time = sum(self.data_info == 'TIME')
        self.n_num = sum(self.data_info == 'NUM')
        self.n_cat = sum(self.data_info == 'CAT')
        self.total_samples = raw_x.shape[0]

        self.cat_col = list(np.where(self.data_info == 'CAT')[0])
        self.num_col = list(np.where(self.data_info == 'NUM')[0])
        self.time_col = list(np.where(self.data_info == 'TIME')[0])

        print('#TIME features: {}'.format(self.n_time))
        print('#NUM features: {}'.format(self.n_num))
        print('#CAT features: {}'.format(self.n_cat))
        
        # Convert sparse to dense if needed
        raw_x = raw_x.toarray() if type(raw_x) == scipy.sparse.csr.csr_matrix else raw_x

        self.pipeline = Pipeline([
            ('label_encoder', CatEncoder(selected_columns=self.cat_col)),
            ('imputer', Imputation(selected_columns=self.cat_col + self.num_col + self.time_col)),
            ('scaler', TabScaler(selected_columns=self.num_col)),
            ('pca', TabPCA(selected_columns=self.num_col)),
            ('time_diff', TimeDiff(selected_columns=self.time_col)),
            ('filter', FilterConstant(selected_columns=self.time_col + self.num_col + self.cat_col)),
            ('pearson_corr', FeatureFilter(selected_columns=self.time_col + self.num_col + self.cat_col)),
            ('lgbm_feat_selection', FeatureImportance(selected_columns=self.time_col + self.num_col + self.cat_col)),
        ])
        self.pipeline.fit(raw_x, y)

        return self

    def transform(self, raw_x, time_limit=None):
        """
        This function should train the model parameters.

        Args:
            raw_x: a numpy.ndarray instance containing the training/testing data.
            time_limit: remaining time budget.
        Both inputs X and y are numpy arrays.
        If fit is called multiple times on incremental data (train, test1, test2, etc.)
        you should warm-start your training from the pre-trained model. Past data will
        NOT be available for re-training.
        """
        # Get Meta-Feature
        if time_limit is None:
            if self.budget is None:
                time_limit = 24 * 60 * 60
                self.budget = time_limit
        else:
            self.budget = time_limit

        # Convert sparse to dense if needed
        raw_x = raw_x.toarray() if type(raw_x) == scipy.sparse.csr.csr_matrix else raw_x
        return self.pipeline.transform(raw_x)

    @staticmethod
    def extract_data_info(raw_x):
        """
        This function extracts the data info automatically based on the type of each feature in raw_x.

        Args:
            raw_x: a numpy.ndarray instance containing the training data.
        """
        data_info = []
        row_num, col_num = raw_x.shape
        for col_idx in range(col_num):
            try:
                raw_x[:, col_idx].astype(np.float)
                data_info.append('NUM')
            except:
                data_info.append('CAT')
        return np.array(data_info)


class TabularData:
    def __init__(self, data, data_info):
        self.data = data
        self.data_info = data_info

        self.total_samples = 0

        self.cat_to_int_label = {}
        self.n_first_batch_keys = {}
        self.high_level_cat_keys = []

        self.num_cat_pair = {}
        self.feature_add_high_cat = 0
        self.feature_add_cat_num = 10
        self.feature_add_cat_cat = 10
        self.order_num_cat_pair = {}

        self.selected_cols = None

        self.n_time = None
        self.n_num = None
        self.n_cat = None


class Primitive(BaseEstimator, TransformerMixin):
    def __init__(self, selected_columns=[], selected_type=None):
        self.selected = selected_columns
        self.selected_type = selected_type

    def fit(self, X, y=None):
        if not self.selected:
            return self
        return self._fit(X, y)

    def transform(self, X, y=None):
        if not self.selected:
            return X
        return self._transform(X, y)

    @abstractmethod
    def _fit(self, X, y=None):
        pass

    @abstractmethod
    def _transform(self, X, y=None):
        pass


class TabScaler(Primitive):
    scaler = None

    def _fit(self, X, y=None):
        self.scaler = StandardScaler()
        self.scaler.fit(X[:, self.selected], y)
        return self

    def _transform(self, X, y=None):
        X[:, self.selected] = self.scaler.transform(X[:, self.selected])
        return X


class BoxCox(Primitive):
    transformer = None

    def _fit(self, X, y=None):
        self.transformer = PowerTransformer()
        self.transformer.fit(X[:, self.selected], y)
        return self

    def _transform(self, X, y=None):
        X[:, self.selected] = self.transformer.transform(X[:, self.selected])
        return X


class Binning(Primitive):
    binner = None

    def __init__(self, selected_columns=[], selected_type=None, strategy='quantile', encoding='ordinal'):
        super().__init__(selected_columns, selected_type)
        self.strategy = strategy
        self.encoding = encoding

    def _fit(self, X, y=None):
        self.binner = KBinsDiscretizer(strategy=self.strategy, encode=self.encoding)
        self.binner.fit(X[:, self.selected], y)
        return self

    def _transform(self, X, y=None):
        X[:, self.selected] = self.binner.transform(X[:, self.selected])
        return X


class CatEncoder(Primitive):
    cat_to_int_label = {}

    def _fit(self, X, y=None):
        for col_index in self.selected:
            self.cat_to_int_label[col_index] = self.cat_to_int_label.get(col_index, {})
            for row_index in range(len(X)):
                key = str(X[row_index, col_index])
                if key not in self.cat_to_int_label[col_index]:
                    self.cat_to_int_label[col_index][key] = len(self.cat_to_int_label[col_index])
        return self

    def _transform(self, X, y=None):
        for col_index in self.selected:
            for row_index in range(len(X)):
                key = str(X[row_index, col_index])
                X[row_index, col_index] = self.cat_to_int_label[col_index].get(key, np.nan)
        return X


class TargetEncoder(Primitive):
    cat_to_int_label = {}

    def _fit(self, X, y=None):
        for col_index in self.selected:
            self.cat_to_int_label[col_index] = self.cat_to_int_label.get(col_index, {})
            for row_index in range(len(X)):
                key = str(X[row_index, col_index])
                if key not in self.cat_to_int_label[col_index]:
                    self.cat_to_int_label[col_index][key] = len(self.cat_to_int_label[col_index])
        return self

    def _transform(self, X, y=None):
        for col_index in self.selected:
            for row_index in range(len(X)):
                key = str(X[row_index, col_index])
                X[row_index, col_index] = self.cat_to_int_label[col_index].get(key, np.nan)
        return X


class FilterConstant(Primitive):
    selected_cols = []

    def _fit(self, X, y=None):
        self.selected_cols = np.where(np.max(X, axis=0) - np.min(X, axis=0) != 0)[0]
        return self

    def _transform(self, X, y=None):
        return X[:, self.selected_cols]


class TimeDiff(Primitive):

    def _fit(self, X, y=None):
        return self

    def _transform(self, X, y=None):
        x_time = X[:, self.selected]
        len_cols = x_time.shape[1]
        for i in range(len_cols):
            for j in range(i + 1, len_cols):
                X = np.append(X, np.expand_dims(x_time[:, i] - x_time[:, j], 1), 1)
        return X


class TimeOffset(Primitive):
    start_time = None

    def _fit(self, X, y=None):
        self.start_time = np.min(X[self.selected], axis=0)
        return self

    def _transform(self, X, y=None):
        X[self.selected] = X[self.selected] - self.start_time
        return X


class TabPCA(Primitive):
    pca = None

    def _fit(self, X, y=None):
        self.pca = PCA(n_components=0.99, svd_solver='full')
        self.pca.fit(X[:, self.selected])
        return self

    def _transform(self, X, y=None):
        x_pca = self.pca.transform(X[:, self.selected])
        return np.concatenate([X, x_pca], axis=1)


class CatCount(Primitive):
    count_dict = {}

    def _fit(self, X, y=None):
        for col in self.selected:
            self.count_dict[col] = collections.Counter(X[:, col])
        return self

    def _transform(self, X, y=None):
        for col in self.selected:
            gen_freq = np.vectorize(lambda key: self.count_dict[col][key])
            X[:, col] = gen_freq(X[:, col])
        return X


class LogTransform(Primitive):

    def _fit(self, X, y=None):
        return self

    def _transform(self, X, y=None):
        for col in self.selected:
            X[:, col] = np.square(np.log(X[:, col]))
        return X


class Imputation(Primitive):
    imputer = None

    def _fit(self, X, y=None):
        # TODO implement most_frequent
        self.imputer = SimpleImputer(strategy='constant', fill_value=0)
        self.imputer.fit(X)
        return self

    def _transform(self, X, y=None):
        return self.imputer.transform(X)


class FeatureFilter(Primitive):
    def __init__(self, selected_columns=[], selected_type=None, threshold=0.001):
        super().__init__(selected_columns, selected_type)
        self.threshold = threshold
        self.drop_columns = []

    def _fit(self, X, y=None):
        for col in self.selected:
            mu = abs(pearsonr(X[:, col], y)[0])
            if np.isnan(mu):
                mu = 0
            if mu < self.threshold:
                self.drop_columns.append(col)
        return self

    def _transform(self, X, y=None):
        X = np.delete(X, self.drop_columns, axis=1)
        return X


class FeatureImportance(Primitive):
    def __init__(self, selected_columns=[], selected_type=None, threshold=0.001, task_type='classification'):
        super().__init__(selected_columns, selected_type)
        self.threshold = threshold
        self.drop_columns = []
        self.task_type = task_type

    def _fit(self, X, y=None):
        if self.task_type == 'classification':
            n_classes = len(set(y))
            if n_classes == 2:
                estimator = LGBMClassifier(silent=False,
                                           verbose=-1,
                                           n_jobs=1,
                                           objective='binary')
            else:
                estimator = LGBMClassifier(silent=False,
                                           verbose=-1,
                                           n_jobs=1,
                                           num_class=n_classes,
                                           objective='multiclass')
        elif self.task_type == 'regression':
            estimator = LGBMRegressor(silent=False,
                                      verbose=-1,
                                      n_jobs=1,
                                      objective='regression')
        estimator.fit(X, y)
        feature_importance = estimator.feature_importances_
        feature_importance = feature_importance/feature_importance.mean()
        self.drop_columns = np.where(feature_importance < self.threshold)[0]
        return self

    def _transform(self, X, y=None):
        X = np.delete(X, self.drop_columns, axis=1)
        return X


if __name__ == "__main__":
    ntime, nnum, ncat = 4, 10, 8
    nsample = 1000
    x_num = np.random.random([nsample, nnum])
    x_time = np.random.random([nsample, ntime])
    x_cat = np.random.randint(0, 10, [nsample, ncat])

    x_all = np.concatenate([x_num, x_time, x_cat], axis=1)
    x_train = x_all[:int(nsample * 0.8), :]
    x_test = x_all[int(nsample * 0.8):, :]

    y_all = np.random.randint(0, 2, nsample)
    y_train = y_all[:int(nsample * 0.8)]
    y_test = y_all[int(nsample * 0.8):]

    datainfo = np.array(['TIME'] * ntime + ['NUM'] * nnum + ['CAT'] * ncat)
    print(x_train[:4, 20])
    prep = TabularPreprocessor()
    prep.fit(x_train, y_train, 24*60*60, datainfo)
    x_new = prep.transform(x_train)

    print("-----")
    print(x_new[:4, 2])

