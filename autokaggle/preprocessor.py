import numpy as np
import pandas as pd
import scipy
import itertools
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PowerTransformer, \
    KBinsDiscretizer, OneHotEncoder
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from abc import abstractmethod
import collections
from lightgbm import LGBMClassifier, LGBMRegressor

LEVEL_HIGH = 32


class Preprocessor(TransformerMixin):
    """ Implements basic preprocessing and feature engineering class.

        Preprocessor takes care of the basic preprocessing and feature engineering of
        the input data. Similar to Scikit-learn transformers,it implements the fit()
        and transform() methods. TO acheive this It applies various feature
        primitives in a sequence using scikit-learn pipeline.
        # Arguments
            config: Config. Defines the configuration of various components of the
            AutoML pipeline.
            params: Dict. Hyper-parameter search space for preprocessor.
            pipeline: Pipeline. Sci-kit learn pipeline class to apply the feature
            primitives in sequence
    """

    def __init__(self, config, params):
        self.config = config
        self.params = params
        self.pipeline = None

    def fit(self, raw_x, y):
        """ This function trains the preprocessor chain
        # Arguments
            raw_x: A numpy array instance containing the training data data.
            y: A numpy array instance containing training label vector.
        # Returns
            None
        This function fits the preprocessor chain on the given training data
        """
        data = TabularData(raw_x, self.config.data_info, self.config.verbose)

        steps = []
        steps.extend(self.get_imputation_pipeline(self.params))
        steps.extend(self.get_higher_order_pipeline(self.params))
        steps.extend(self.get_categorical_pipeline(self.params))
        steps.extend(self.get_numerical_pipeline(self.params))
        steps.extend(self.get_time_pipeline(self.params))
        steps.extend(self.get_filtering_pipeline(self.params))
        self.pipeline = Pipeline(steps)

        self.pipeline.fit(data, y)

        return self

    def transform(self, raw_x):
        """ Generate data transformation on the given data.
        # Arguments
            raw_x: a numpy array instance containing the training/testing data
        # Returns
            A numpy array instance containing the transformed data.
        This function provides transforms the input data by applying the
        transformations using the pre-trained preprocessor chain.
        """
        # Get Meta-Feature
        data = TabularData(raw_x, self.config.data_info, self.config.verbose)
        a = self.pipeline.transform(data).X
        return a.values

    @staticmethod
    def get_categorical_pipeline(params):
        """ Generate pipeline of primitives for categorical features.
        # Arguments
            params: Hyper-parameter setting for the preprocessors.
        # Returns
            List of primitives to be applied (based on the given setting)
        """
        choice = params.get('cat_encoding', 'target')
        cat_pipeline = []
        if choice == 'target':
            cat_pipeline.append(('target_encoder', TargetEncoder(operation='upd',
                                                                 selected_type='CAT')
                                 ))
        elif choice == 'label':
            cat_pipeline.append(
                ('label_encoder', LabelEncode(operation='upd', selected_type='CAT')))
        elif choice == 'count':
            cat_pipeline.append(
                ('count_encoder', CatCount(operation='upd', selected_type='CAT')))
        elif choice == 'target+count':
            cat_pipeline.append(('target_encoder', TargetEncoder(operation='add',
                                                                 selected_type='CAT')
                                 ))
            cat_pipeline.append(
                ('count_encoder', CatCount(operation='upd', selected_type='CAT')))
        elif choice == 'one_hot':
            cat_pipeline.append(
                ('one_hot_encoder', OneHot(operation='upd', selected_type='CAT')))
        elif choice == 'target+label':
            cat_pipeline.append(('target_encoder', TargetEncoder(operation='add',
                                                                 selected_type='CAT')
                                 ))
            cat_pipeline.append(
                ('label_encoder', LabelEncode(operation='upd', selected_type='CAT')))
        else:
            raise ValueError
        return cat_pipeline

    @staticmethod
    def get_numerical_pipeline(params):
        """ Generate pipeline of primitives for numerical features.
        # Arguments
            params: Hyper-parameter setting for the preprocessors.
        # Returns
            List of primitives to be applied (based on the given setting)
        """
        scaling = params.get('scaling', True)
        log_transform = params.get('log_transform', False)
        power_transform = params.get('power_transform', False)
        pca = params.get('pca', False)
        binning = params.get('binning', False)

        numeric_pipeline = []
        if scaling:
            numeric_pipeline.append(
                ('scaler', TabScaler(operation='upd', selected_type='NUM')))
        if log_transform:
            numeric_pipeline.append(('log_transform',
                                     LogTransform(operation='upd',
                                                  selected_type='NUM')))
        if power_transform:
            numeric_pipeline.append(
                ('boxcox', BoxCox(operation='upd', selected_type='NUM')))
        if pca:
            numeric_pipeline.append(
                ('pca', TabPCA(operation='add', selected_type='NUM')))
        if binning:
            numeric_pipeline.append(
                ('binning', Binning(operation='add', selected_type='NUM')))
        return numeric_pipeline

    def get_filtering_pipeline(self, params):
        """ Generate pipeline of primitives to filter less useful features.
        # Arguments
            params: Hyper-parameter setting for the preprocessors.
        # Returns
            List of primitives to be applied (based on the given setting)
        """
        pearson_thresh = params.get('pearson_thresh', 0)
        feat_importance_thresh = params.get('feat_importance_thresh', 0)

        filter_pipeline = [
            ('filter', FilterConstant(operation='del', selected_type='ALL'))]
        if pearson_thresh > 0:
            filter_pipeline.append(
                ('pearson_corr', FeatureFilter(operation='del', selected_type='ALL',
                                               threshold=pearson_thresh)))
        if feat_importance_thresh > 0:
            filter_pipeline.append(
                ('lgbm_feat_selection',
                 FeatureImportance(operation='del',
                                   selected_type='ALL',
                                   threshold=feat_importance_thresh,
                                   task_type=self.config.objective)))
        return filter_pipeline

    @staticmethod
    def get_time_pipeline(params):
        """ Generate pipeline of primitives for time features.
        # Arguments
            params: Hyper-parameter setting for the preprocessors.
        # Returns
            List of primitives to be applied (based on the given setting)
        """
        add_offset = params.get('add_time_offset', False)
        add_diff = params.get('add_time_diff', False)
        time_pipeline = []
        if add_offset:
            time_pipeline.append(
                ('time_offset', TimeOffset(operation='upd', selected_type='TIME')))
        if add_diff:
            time_pipeline.append(
                ('time_diff', TimeDiff(operation='add', selected_type='TIME')))
        return time_pipeline

    @staticmethod
    def get_imputation_pipeline(params):
        """ Generate pipeline of primitives to impute the missing values.
        # Arguments
            params: Hyper-parameter setting for the preprocessors.
        # Returns
            List of primitives to be applied (based on the given setting)
        """
        strategy = params.get('imputation_strategy', 'most_frequent')
        impute_pipeline = [('imputer',
                            Imputation(operation='upd', selected_type='ALL',
                                       strategy=strategy))]
        return impute_pipeline

    @staticmethod
    def get_higher_order_pipeline(params):
        """ Generate pipeline of primitives to generate cross-column features.
        # Arguments
            params: Hyper-parameter setting for the preprocessors.
        # Returns
            List of primitives to be applied (based on the given setting)
        """
        cat_num_strategy = params.get('cat_num_strategy', None)
        cat_cat_strategy = params.get('cat_cat_strategy', None)
        pipeline = []
        if cat_num_strategy:
            pipeline.append(('cat_num_encoder',
                             CatNumEncoder(operation='add', selected_type1='CAT',
                                           selected_type2='NUM',
                                           strategy=cat_num_strategy)))
        if cat_cat_strategy:
            pipeline.append(('cat_cat_encoder',
                             CatCatEncoder(operation='add', selected_type1='CAT',
                                           selected_type2='CAT',
                                           strategy=cat_cat_strategy)))
        return pipeline


class TabularData:
    """ Represents the data and its meta-info.

        TabularData includes the training/testing data along with its meta info such
        as data types, cardinality etc. The user can update the data and its meta
        info as well as select the features matching the criteria.
        # Arguments
            verbose: Bool. Determines the verbosity of the logging.
            data_info: Dict. Dictionary mapping the feature names to their data_types
            total_samples: Int. Number of samples in the data
            cat_col: List. List of the categorical features
            num_col: List. List of the numerical features
            time_col: List. List of the time features
            n_cat: Int. Number of categorical features
            n_num: Int. Number of numerical features
            n_time: Int. Number of time features
            cat_cardinality: Dict. Dictionary mapping categorical feature names of
            their cardinality (no. of unique values)
            generated_features: List. List of the newly added features. (In
            addition to the pre-existing columns)
            num_info: Dict. Dictionary mapping numeircal column to their meta info
            such as range, std etc.
    """

    def __init__(self, raw_x, data_info, verbose=True):
        self.cat_col = None
        self.num_col = None
        self.time_col = None
        self.n_cat = 0
        self.n_time = 0
        self.n_num = 0
        self.cat_cardinality = None
        self.generated_features = None
        self.num_info = None
        self.verbose = verbose
        self.data_info = {str(i): data_info[i] for i in range(len(data_info))}
        self.total_samples = raw_x.shape[0]
        self.refresh_col_types()

        # Convert sparse to dense if needed
        raw_x = raw_x.toarray() if type(
            raw_x) == scipy.sparse.csr.csr_matrix else raw_x

        # To pandas Dataframe
        if type(raw_x) != pd.DataFrame:
            raw_x = pd.DataFrame(raw_x,
                                 columns=[str(i) for i in range(raw_x.shape[1])])

        self.X = raw_x
        # self.update_cat_cardinality()

    def update_type(self, columns, new_type):
        """ Updates the column datatype.
        # Arguments
            column: List of columns whose data_type needs update.
            new_type: New data_type (either of 'CAT', 'NUM' or 'TIME').
        # Returns
            None.
        This function updates the data types of given list of columns.
        """
        for c in columns:
            self.data_info[c] = new_type

    def delete_type(self, columns):
        """ Delete the columns from the feature to data_type mapping.
        # Arguments
            column: List of columns whose data_type needs update.
        # Returns
            None
        This function removes the selected columns from the data_info dictionary.
        """
        for c in columns:
            _ = self.data_info.pop(c, 0)

    def rename_cols(self, key):
        """ Provides a rename function to add new columns without collision.
        # Arguments
            key: Identifier for renaming
        # Returns
            Renaming function which takes current column name and outputs a new
            unique column name.
        """

        def rename_fn(col_name):
            col_name = str(col_name)
            col_name += '_' + key
            while col_name in self.X.columns:
                col_name += '_' + key
            return col_name

        return rename_fn

    def update(self, operation, columns, x_tr, new_type=None, key=''):
        """ Updates the TabularData after applying primitive.
        # Arguments
            operation: Primitive operation applied ('add', 'update' or 'delete').
            columns: List of columns affected.
            x_tr: Transformed (or newly generated) features
            new_type: Data type of the new column
            key: Name key for renaming the new columns
        # Returns
            None
        This function takes the transformed (or generated) features after applying
        the primitive and updates the
        TabularData.
        """
        if operation == 'upd':
            if x_tr is not None:
                self.X[columns] = x_tr
            if new_type is not None:
                self.update_type(columns, new_type)
        elif operation == 'add':
            if x_tr is not None:
                x_tr = x_tr.rename(columns=self.rename_cols(key))
                self.X = pd.concat([self.X, x_tr], axis=1)
                self.update_type(x_tr.columns, new_type)
        elif operation == 'del':
            if len(columns) != 0:
                self.X.drop(columns=columns, inplace=True)
                self.delete_type(columns)
        else:
            print("invalid operation")
        self.refresh_col_types()

    def refresh_col_types(self):
        """ Updates the column_types based on the data_info
        # Arguments
            None
        # Returns
            None
        This function updates the cat, num and time column lists based on (any)
        updates in the data_info.
        """
        self.cat_col = [k for k, v in self.data_info.items() if v == 'CAT']
        self.num_col = [k for k, v in self.data_info.items() if v == 'NUM']
        self.time_col = [k for k, v in self.data_info.items() if v == 'TIME']
        self.n_time = len(self.time_col)
        self.n_num = len(self.num_col)
        self.n_cat = len(self.cat_col)

    def update_cat_cardinality(self):
        """ Update categorical cardinality mapping for all categorical columns.
        # Arguments
            None
        # Returns
            None
        """
        # TODO: too slow make it faster
        if not self.cat_cardinality:
            self.cat_cardinality = {}
        for c in self.cat_col:
            self.cat_cardinality[c] = len(set(self.X[c]))

    def select_columns(self, data_type):
        """ Returns all the columns matching the input data_type
        # Arguments
            data_type: Required type of the data (either of 'CAT', 'NUM', 'TIME' or
            'ALL')
        # Returns
            List of the feature columns matching the input criteria.
        """
        self.refresh_col_types()
        if data_type == 'CAT':
            return self.cat_col
        elif data_type == 'TIME':
            return self.time_col
        elif data_type == 'NUM':
            return self.num_col
        elif data_type == 'ALL':
            return list(self.data_info.keys())
        else:
            print('invalid Type')
            return []


class Primitive(BaseEstimator, TransformerMixin):
    """ Base class for the single order data transformation function.

        Primitive learns and applies the data transformation on a given set of
        features. The user can use fit() and transform() functions to apply these
        transformations.

        # Arguments
            options: Dict. Special arguments specific to the given primitive.
            selected_type: 'String'. Specifies the type of features the
            transformation is supposed to be applied to.
            operation: 'String'. Specifies the type of operation from 'add', 'update'
             or 'delete'
            name_key : 'String'. Signature key to rename the column after applying
            the primitive.
            selected: 'List'. List of the selected features, on which the
            transformation will be applied
            drop_columns: 'List'. List of the features which would be dropped after
            applying the transformation.
            supported_ops: Tuple. Specifies the allowed list of operations for this
            primitive.
    """

    def __init__(self, operation='upd', selected_type=None, **kwargs):
        self.options = None
        self.selected = None
        self.drop_columns = None
        self.supported_ops = ('add', 'upd', 'del')
        self.selected_type = selected_type
        self.operation = operation
        self.init_vars(**kwargs)
        self.name_key = self.__class__.__name__

    def init_vars(self, **kwargs):
        """ Initialize the primitive specific variables (which are not defined in the
        base class)
        # Arguments
            kwargs: Dictionary containing primitive specific variables
        # Returns
            None.
        """
        self.options = kwargs

    def fit(self, data, y=None):
        """ A wrapper function to train the given primitive on the input training
        data.
        # Arguments
            data: A TabularData instance of training data.
            y: A numpy array of the target values.
        # Returns
            None
        """
        self.selected = data.select_columns(self.selected_type)
        if self.operation not in self.supported_ops:
            print("Operation {} not supported for {}".format(self.operation,
                                                             self.__class__.__name__)
                  )
            self.selected = None
        if not self.selected:
            return self
        return self._fit(data, y)

    def transform(self, data, y=None):
        """ A wrapper function to generate transformation on the input data based on
        pre-trained primitive.
        # Arguments
            data: Input training/testing data in TabularData form.
            y: A numpy array of the target values.
        # Returns
            A TabularData instance of the transformed data.
        """
        if not self.selected:
            return data
        return self._transform(data, y)

    @abstractmethod
    def _fit(self, data, y=None):
        """ Contains the actual implementation of training the primitive (implemented
        in the child class)
        # Arguments
            data: A TabularData instance of training data.
            y: A numpy array of the target values.
        # Returns
            None
        """
        pass

    @abstractmethod
    def _transform(self, data, y=None):
        """ Contains the actual implementation of transforming the data using
        primitive. (implemented in the child class)
        # Arguments
            data: Input training/testing data in TabularData form.
            y: A numpy array of the target values.
        # Returns
            A TabularData instance of the transformed data.
        """
        pass


class PrimitiveHigherOrder:
    """ Base class for the cross-order data transformation function.

        PrimitiveHigherOrder learns and applies the data transformation across two
        sets of features. The user can use fit() and transform() functions to
        apply these transformations.

        # Arguments
            options: Dict. Special arguments specific to the given primitive.
            selected_type1: 'String'. Specifies the first type of features the
            transformation is supposed to be applied to.
            selected_type2: 'String'. Specifies the second type of features the
            transformation is supposed to be applied to.
            operation: 'String'. Specifies the type of operation from 'add', 'update'
             or 'delete'
            name_key : 'String'. Signature key to rename the column after applying
            the primitive.
            selected_1: 'List'. List of the selected features in the first set, on
            which the transformation will be
            applied
            selected_2: 'List'. List of the selected features in the second set, on
            which the transformation will be
            applied
            drop_columns: 'List'. List of the features which would be dropped after
            applying the transformation.
            supported_ops: Tuple. Specifies the allowed list of operations for this
            primitive.
    """

    def __init__(self, operation='upd', selected_type1=None, selected_type2=None,
                 **kwargs):
        self.options = None
        self.selected_1 = None
        self.selected_2 = None
        self.drop_columns = None
        self.supported_ops = ('add', 'upd', 'del')
        self.operation = operation
        self.selected_type1 = selected_type1
        self.selected_type2 = selected_type2
        self.init_vars(**kwargs)
        self.name_key = self.__class__.__name__

    def init_vars(self, **kwargs):
        """ Initialize the primitive specific variables (which are not defined in the
        base class)
        # Arguments
            kwargs: Dictionary containing primitive specific variables
        # Returns
            None.
        """
        self.options = kwargs

    def fit(self, data, y=None):
        """ A wrapper function to train the given primitive on the input training
        data.
        # Arguments
            data: A TabularData instance of training data.
            y: A numpy array of the target values.
        # Returns
            None
        """
        self.selected_1 = data.select_columns(self.selected_type1)
        self.selected_2 = data.select_columns(self.selected_type2)

        if self.operation not in self.supported_ops:
            print("Operation {} not supported for {}".format(self.operation,
                                                             self.__class__.__name__)
                  )
            self.selected_1 = None
            self.selected_2 = None
        if not self.selected_1 or not self.selected_2:
            return self
        return self._fit(data, y)

    def transform(self, data, y=None):
        """ A wrapper function to generate transformation on the input data based on
        pre-trained primitive.
        # Arguments
            data: Input training/testing data in TabularData form.
            y: A numpy array of the target values.
        # Returns
            A TabularData instance of the transformed data.
        """
        if not self.selected_1 or not self.selected_2:
            return data
        return self._transform(data, y)

    @abstractmethod
    def _fit(self, data, y=None):
        """ Contains the actual implementation of training the primitive (implemented
        in the child class)
        # Arguments
            data: A TabularData instance of training data.
            y: A numpy array of the target values.
        # Returns
            None
        """
        pass

    @abstractmethod
    def _transform(self, data, y=None):
        """ Contains the actual implementation of transforming the data using
        primitive. (implemented in the child class)
        # Arguments
            data: Input training/testing data in TabularData form.
            y: A numpy array of the target values.
        # Returns
            A TabularData instance of the transformed data.
        """
        pass


class TabScaler(Primitive):
    """ Standard Scaler primitive.

        TabScaler scales the selected numerical features to have 0 mean and unit
        variance.

        # Arguments
            scaler: StandardScaler. Instance of scikit-learn StandardScaler object
    """
    scaler = None
    supported_ops = ('add', 'upd')

    def _fit(self, data, y=None):
        self.scaler = StandardScaler()
        self.scaler.fit(data.X[self.selected], y)
        return self

    def _transform(self, data, y=None):
        x_tr = self.scaler.transform(data.X[self.selected])
        data.update(self.operation, self.selected, x_tr, new_type='NUM',
                    key=self.name_key)
        return data


class BoxCox(Primitive):
    """ Power Transform primitive.

        The class applies BoxCox power transformation to make the selected features
        have normal distribution.

        # Arguments
            transformer: PowerTransformer. Instance of scikit-learn PowerTransformer
            object
    """
    transformer = None
    supported_ops = ('add', 'upd')

    def _fit(self, data, y=None):
        self.transformer = PowerTransformer()
        self.transformer.fit(data.X[self.selected], y)
        return self

    def _transform(self, data, y=None):
        x_tr = self.transformer.transform(data.X[self.selected])
        data.update(self.operation, self.selected, x_tr, new_type='NUM',
                    key=self.name_key)
        return data


class Binning(Primitive):
    """ Numerical binning primitive.

        The class applies divides the given numeric column in the list of buckets,
        based on the range of their values.

        # Arguments
            binner: KBinsDiscretizer. Instance of scikit-learn KBinsDiscretizer
            object
            strategy: String. Strategy used to define width of the bins. Possible
            options are: (‘uniform’, ‘quantile’,
            ‘kmeans’)
            encoding: String. Method used to encode the transformed result. Possible
            options are: (‘onehot’,
            ‘onehot-dense’, ‘ordinal’)
    """
    binner = None
    strategy = None
    encoding = None
    supported_ops = ('add', 'upd')

    def init_vars(self, strategy='quantile', encoding='ordinal'):
        self.strategy = strategy
        self.encoding = encoding

    def _fit(self, data, y=None):
        self.binner = KBinsDiscretizer(strategy=self.strategy, encode=self.encoding)
        self.binner.fit(data.X[self.selected], y)
        return self

    def _transform(self, data, y=None):
        x_tr = pd.DataFrame(self.binner.transform(data.X[self.selected]))
        data.update(self.operation, self.selected, x_tr, new_type='NUM',
                    key=self.name_key)
        return data


class OneHot(Primitive):
    """ One Hot Encoder for categorical features.

        The class applies one hot encoding to categorical features, using the
        sklearn implementation.

        # Arguments
            ohe: OneHotEncoder. Instance of scikit-learn OneHotEncoder object
    """
    ohe = None
    supported_ops = ('add', 'upd')

    def _fit(self, data, y=None):
        self.ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.ohe.fit(data.X[self.selected], y)
        return self

    def _transform(self, data, y=None):
        x_tr = pd.DataFrame(self.ohe.transform(data.X[self.selected]))
        if self.operation == 'add':
            data.update(self.operation, self.selected, x_tr, new_type='NUM',
                        key=self.name_key)
        elif self.operation == 'upd':
            data.update('add', self.selected, x_tr, new_type='NUM',
                        key=self.name_key)
            data.update('del', self.selected, None, None, key=self.name_key)
        return data


class LabelEncode(Primitive):
    """ Label Encoder for categorical features.

        The class applies Label Encoding to categorical features, By mapping each
        category to a numerical value.

        # Arguments
            cat_to_int_label: Dict. Mapping from categories to their assigned integer
            value
            unknown_key_dict: Dict. Mapping for each categorical feature column to
            the integer value to replace the previously unseen categories
    """
    cat_to_int_label = None
    unknown_key_dict = None
    supported_ops = ('add', 'upd')

    def _fit(self, data, y=None):
        self.cat_to_int_label = {}
        self.unknown_key_dict = {}
        for col in self.selected:
            self.cat_to_int_label[col] = {key: idx for idx, key in
                                          enumerate(set(data.X[col]))}
            self.unknown_key_dict[col] = len(self.cat_to_int_label[col])
        return self

    def _transform(self, data, y=None):
        x_tr = pd.DataFrame()
        for col in self.selected:
            x_tr[col] = data.X[col].apply(
                lambda key: self.cat_to_int_label[col].get(key,
                                                           self.unknown_key_dict[
                                                               col]))
        data.update(self.operation, self.selected, x_tr, new_type='NUM',
                    key=self.name_key)
        return data


class TargetEncoder(Primitive):
    """ Target Encoder for categorical features.

        The class applies target encoding to categorical features, By learning
        the mapping of category to numeric value
        based on some aggregation of the target value.

        # Arguments
            target_encoding_map: Dict. Mapping from categories to their assigned
            numeric value
    """
    target_encoding_map = None
    supported_ops = ('add', 'upd')

    @staticmethod
    def calc_smooth_mean(df, by, on, alpha=5):
        """ Calculates the smoothed means on the target value.
        # Arguments
            df: Input dataframe
            by: Groupby column (categorical column)
            on: Target column
            alpha: smoothing factor
        # Returns
            smoothed mean and the overall mean
        """
        # Compute the global mean
        mean = df[on].mean()

        # Compute the number of values and the mean of each group
        agg = df.groupby(by)[on].agg(['count', 'mean'])
        counts = agg['count']
        means = agg['mean']

        # Compute the "smoothed" means
        smooth = (counts * means + alpha * mean) / (counts + alpha)
        return smooth, mean

    def _fit(self, data, y=None):
        X = data.X
        self.target_encoding_map = {}
        X['target'] = y
        for col in self.selected:
            self.target_encoding_map[col] = self.calc_smooth_mean(X, col, 'target',
                                                                  alpha=5)
        X.drop('target', axis=1, inplace=True)
        return self

    def _transform(self, data, y=None):
        x_tr = pd.DataFrame()
        for col in self.selected:
            x_tr[col] = data.X[col].map(self.target_encoding_map[col][0],
                                        self.target_encoding_map[col][1])
        data.update(self.operation, self.selected, x_tr, new_type='NUM',
                    key=self.name_key)
        return data


class CatCatEncoder(PrimitiveHigherOrder):
    """ Cross column feature generator between categorical and categorical columns.

        The class learns a new features based on the values of selected two
        categorical features.

        # Arguments
            cat_cat_map: Dict. Mapping from cat-cat combination to the an assigned
            numeric value
            strategy: String. Aggregation strategy to learn the mapping between
            cat-cat combination to numeric value
    """
    supported_ops = ('add',)
    cat_cat_map = None
    strategy = None

    def init_vars(self, strategy='count'):
        self.strategy = strategy

    @staticmethod
    def cat_cat_count(df, col1, col2, strategy='count'):
        """ Generate mapping for cat-cat combination to the numerical value based on
        the given strategy.
        # Arguments
            col1: First categorical column
            col2: Second categorical column
            strategy: Aggregation strategy
        # Returns
            Mapping from cat-cat combination to the numeric value..
        """
        if strategy == 'count':
            mapping = df.groupby([col1])[col2].count()
        elif strategy == 'nunique':
            mapping = df.groupby([col1])[col2].nunique()
        else:
            mapping = df.groupby([col1])[col2].count() // df.groupby([col1])[
                col2].nunique()
        return mapping

    def _fit(self, data, y=None):
        self.cat_cat_map = {}
        self.selected_1 = list(set(self.selected_1 + self.selected_2))
        for col1, col2 in itertools.combinations(self.selected_1, 2):
            self.cat_cat_map[col1 + '_cross_' + col2] = \
                self.cat_cat_count(data.X,
                                   col1,
                                   col2,
                                   self.strategy)
        return self

    def _transform(self, data, y=None):
        x_tr = pd.DataFrame()
        for col1, col2 in itertools.combinations(self.selected_1, 2):
            if col1 + '_cross_' + col2 in self.cat_cat_map:
                x_tr[col1 + '_cross_' + col2] = data.X[col1].map(
                    self.cat_cat_map[col1 + '_cross_' + col2])
        data.update(self.operation, self.selected_1, x_tr, new_type='NUM',
                    key=self.name_key)
        return data


class CatNumEncoder(PrimitiveHigherOrder):
    """ Cross column feature generator between categorical and numerical columns.

        The class learns a new features based on the values of selected categorical
        and numerical features.

        # Arguments
            cat_num_map: Dict. Mapping from cat-num combination to the an assigned
            numeric value
            strategy: String. Aggregation strategy to learn the mapping between
            cat-num combination to numeric value
    """
    supported_ops = ('add',)
    cat_num_map = None
    strategy = None

    def init_vars(self, strategy='mean'):
        self.strategy = strategy

    @staticmethod
    def cat_num_interaction(df, col1, col2, method='mean'):
        """ Generate mapping for cat-num combination to the numerical value based on
        the given strategy.
        # Arguments
            col1: categorical column
            col2: numerical column
            method: Aggregation strategy
        # Returns
            Mapping from cat-num combination to the numeric value..
        """
        if method == 'mean':
            mapping = df.groupby([col1])[col2].mean()
        elif method == 'std':
            mapping = df.groupby([col1])[col2].std()
        elif method == 'max':
            mapping = df.groupby([col1])[col2].max()
        elif method == 'min':
            mapping = df.groupby([col1])[col2].min()
        else:
            mapping = df.groupby([col1])[col2].mean()

        return mapping

    def _fit(self, data, y=None):
        self.cat_num_map = {}
        for col1 in self.selected_1:
            for col2 in self.selected_2:
                self.cat_num_map[col1 + '_cross_' + col2] = self.cat_num_interaction(
                    data.X, col1, col2, self.strategy)
        return self

    def _transform(self, data, y=None):
        x_tr = pd.DataFrame()
        for col1 in self.selected_1:
            for col2 in self.selected_2:
                if col1 + '_cross_' + col2 in self.cat_num_map:
                    x_tr[col1 + '_cross_' + col2] = data.X[col1].map(
                        self.cat_num_map[col1 + '_cross_' + col2])
        data.update(self.operation, self.selected_1, x_tr, new_type='NUM',
                    key=self.name_key)
        return data


class CatBinEncoder(PrimitiveHigherOrder):
    """ Cross column feature generator between categorical and binary columns.

        The class learns a new features based on the values of selected categorical
        and binary features.

        # Arguments
            cat_bin_map: Dict. Mapping from cat-bin combination to the an assigned
            numeric value
            strategy: String. Aggregation strategy to learn the mapping between
            cat-bin combination to numeric value
    """
    supported_ops = ('add',)
    cat_bin_map = None
    strategy = None

    def init_vars(self, strategy='percent_true'):
        self.strategy = strategy

    @staticmethod
    def cat_bin_interaction(df, col1, col2, strategy='percent_true'):
        """ Generate mapping for cat-bin combination to the numerical value based on
        the given strategy.
        # Arguments
            col1: Categorical column
            col2: Binary column
            strategy: Aggregation strategy
        # Returns
            Mapping from cat-bin combination to the numeric value..
        """
        if strategy == 'percent_true':
            mapping = df.groupby([col1])[col2].mean()
        elif strategy == 'count':
            mapping = df.groupby([col1])[col2].count()
        else:
            mapping = df.groupby([col1])[col2].mean()
        return mapping

    def _fit(self, data, y=None):
        self.cat_bin_map = {}
        for col1 in self.selected_1:
            for col2 in self.selected_2:
                self.cat_bin_map[col1 + '_cross_' + col2] = self.cat_bin_interaction(
                    data.X, col1, col2, self.strategy)
        return self

    def _transform(self, data, y=None):
        x_tr = pd.DataFrame()
        for col1 in self.selected_1:
            for col2 in self.selected_2:
                if col1 + '_cross_' + col2 in self.cat_bin_map:
                    x_tr[col1 + '_cross_' + col2] = data.X[col1].map(
                        self.cat_bin_map[col1 + '_cross_' + col2])
        data.update(self.operation, self.selected_1, x_tr, new_type='NUM',
                    key=self.name_key)
        return data


class FilterConstant(Primitive):
    """ Filters the constant or very low variance columns.

        The class finds the non-changing or very low variance columns and marked them
        for deletion, so that they are not used by the machine learning estimator.
    """
    drop_columns = None
    supported_ops = ('del',)

    def _fit(self, data, y=None):
        X = data.X[self.selected]
        self.drop_columns = X.columns[(X.max(axis=0) - X.min(axis=0) == 0)].tolist()
        return self

    def _transform(self, data, y=None):
        data.update(self.operation, self.drop_columns, None, new_type=None,
                    key=self.name_key)
        return data


class TimeDiff(Primitive):
    """ Adds features based on difference of time values.

        This class generates the features as time difference between two selected
        time columns.
    """
    supported_ops = ('add',)

    def _fit(self, data, y=None):
        return self

    def _transform(self, data, y=None):
        x_tr = pd.DataFrame()
        for a, b in itertools.combinations(self.selected, 2):
            x_tr[a + '-' + b] = data.X[a] - data.X[b]
        data.update(self.operation, self.selected, x_tr, new_type='TIME',
                    key=self.name_key)
        return data


class TimeOffset(Primitive):
    """ Updates the time features in terms of difference from the start value.

        This class updates the time features such that they are represented as a
        difference from the start time.

        # Arguments
            start_time: Int. Starting time of the selected time feature.
    """
    start_time = None
    supported_ops = ('add', 'upd')

    def _fit(self, data, y=None):
        self.start_time = data.X[self.selected].min(axis=0)
        return self

    def _transform(self, data, y=None):
        x_tr = pd.DataFrame()
        x_tr[self.selected] = data.X[self.selected] - self.start_time
        data.update(self.operation, self.selected, x_tr, new_type='TIME',
                    key=self.name_key)
        return data


class TabPCA(Primitive):
    """ Generates new features by finding PCA of the selected features.

        The class calculates the PCA of the selected features and adds the
        transformation as new set of features.
        # Arguments
            pca: PCA. Scikit-lean PCA class.
    """
    pca = None
    supported_ops = ('add',)

    def _fit(self, data, y=None):
        self.pca = PCA(n_components=0.99, svd_solver='full')
        self.pca.fit(data.X[self.selected])
        return self

    def _transform(self, data, y=None):
        x_pca = self.pca.transform(data.X[self.selected])
        x_pca = pd.DataFrame(x_pca, columns=['pca_' + str(i) for i in
                                             range(x_pca.shape[1])])
        data.update(self.operation, self.selected, x_pca, new_type='NUM',
                    key=self.name_key)
        return data


class CatCount(Primitive):
    """ Count Encoding.

        Replaces the cargorical variables by their occrance count.
        # Arguments
            count_dict: Dict. Mapping of the categories to their respective frequency
            count.
            unknown_key: Float. Mapping value for previously unseen category.
    """
    count_dict = None
    unknown_key = 0
    supported_ops = ('add', 'upd')

    def _fit(self, data, y=None):
        self.count_dict = {}
        for col in self.selected:
            self.count_dict[col] = collections.Counter(data.X[col])
        return self

    def _transform(self, data, y=None):
        x_tr = pd.DataFrame()
        for col in self.selected:
            x_tr[col] = data.X[col].apply(
                lambda key: self.count_dict[col].get(key, self.unknown_key))
        data.update(self.operation, self.selected, x_tr, new_type='NUM',
                    key=self.name_key)
        return data


class LogTransform(Primitive):
    """ Calculates the log transformation.

        The class Calculates the log transform value of the given numeric feature.
        The formula is: sign(x) * log(1 + mod(x))
    """
    name_key = 'log_'
    supported_ops = ('add', 'upd')

    def _fit(self, data, y=None):
        return self

    def _transform(self, data, y=None):
        x_tr = pd.DataFrame()
        for col in self.selected:
            x_tr[self.name_key + col] = np.sign(data.X[col]) * np.log(
                1 + np.abs(data.X[col]))
        data.update(self.operation, self.selected, x_tr, new_type='NUM',
                    key=self.name_key)
        return data


class Imputation(Primitive):
    """ Filters the features based on Pearson Correlation.

        The class removes the features who have low pearson correlation with the
        target.
        # Arguments
            threshold: Float. Threshold for filtering features.
    """
    impute_dict = None
    supported_ops = ('add', 'upd')
    strategy = None

    def init_vars(self, strategy='most_frequent'):
        self.strategy = strategy

    def _fit(self, data, y=None):
        self.impute_dict = {}
        for col in self.selected:
            if self.strategy == 'most_frequent':
                value_counts = data.X[col].value_counts()
                self.impute_dict[
                    col] = value_counts.idxmax() if not value_counts.empty else 0
            elif self.strategy == 'zero':
                self.impute_dict[col] = 0
            else:
                raise ValueError
        return self

    def _transform(self, data, y=None):
        x_tr = pd.DataFrame()
        for col in self.selected:
            x_tr[col] = data.X[col].fillna(self.impute_dict[col])
        data.update(self.operation, self.selected, x_tr, new_type=None,
                    key=self.name_key)
        return data


class FeatureFilter(Primitive):
    """ Filters the features based on Pearson Correlation.

        The class removes the features who have low pearson correlation with the
        target.
        # Arguments
            threshold: Float. Threshold for filtering features.
    """
    threshold = None
    supported_ops = ('del',)

    def init_vars(self, threshold=0.001):
        if threshold == 0:
            self.selected = None
        self.threshold = threshold
        self.drop_columns = []

    def _fit(self, data, y=None):
        for col in self.selected:
            mu = abs(pearsonr(data.X[col], y)[0])
            if np.isnan(mu):
                mu = 0
            if mu < self.threshold:
                self.drop_columns.append(col)
        return self

    def _transform(self, data, y=None):
        data.update(self.operation, self.drop_columns, None, new_type=None,
                    key=self.name_key)
        return data


class FeatureImportance(Primitive):
    """ Filters the features based on feature importance score.

        The class learns a Light GBM estimator for the given data and based on the
        feature importance scores, filters the features with importance lower than
        the threshold.
        # Arguments
            threshold: Float. Threshold for filtering features.
            task_type: 'String'. Specifies the task type amongst: ('classification',
            'regression')
    """
    threshold = None
    task_type = 'classification'
    supported_ops = ('del',)

    def init_vars(self, threshold=0.001, task_type='classification'):
        if threshold == 0:
            self.selected = None
        self.threshold = threshold
        self.drop_columns = []
        self.task_type = task_type

    def _fit(self, data, y=None):
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
        else:
            # self.task_type == 'regression'
            estimator = LGBMRegressor(silent=False,
                                      verbose=-1,
                                      n_jobs=1,
                                      objective='regression')
        estimator.fit(data.X, y)
        feature_importance = estimator.feature_importances_
        feature_importance = feature_importance / feature_importance.mean()
        self.drop_columns = data.X.columns[
            np.where(feature_importance < self.threshold)[0]]
        return self

    def _transform(self, data, y=None):
        data.update(self.operation, self.drop_columns, None, new_type=None,
                    key=self.name_key)
        return data


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
    prep = Preprocessor()
    prep.fit(x_train, y_train, 24 * 60 * 60, datainfo)
    x_new = prep.transform(x_train)

    print("-----")
    print(x_new[:4, 2])
