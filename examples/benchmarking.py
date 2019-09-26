import string
import random
import sys

sys.path.append("../")
import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.datasets
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, f1_score, \
    balanced_accuracy_score, \
    mean_absolute_error, mean_squared_error
# from autosklearn.regression import AutoSklearnRegressor
# from autosklearn.classification import AutoSklearnClassifier
from autokaggle import *
from autokaggle.utils import *
import openml

openml.config.apikey = '3c7196c92a274c3b9405a7e26e9f848e'
import warnings
from abc import abstractmethod
import statistics


def generate_rand_string(size):
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))


class BenchmarkingBase:
    """ Base class for benchmarking autoML platforms.

        This class benchmarks the performance of the given autoML platform. The
        user can call evaluate() method to evaluate the performance on a single
        task or run_automation() for the list of the tasks. The tasks are OpenML
        tasks, which specify the dataset and the train/test/validation folds etc.

        # Arguments
            results: List. List of the results for each evaluation
            sess_name: String. Name of the evaluation session, used for storing
            the results.
            cls_desc: List. List of the columns to be added in classification result
            rgs_desc: List. List of the columns to be added in regression result
            cls_results: DataFrame. Table storing the classification results
            rgs_results: DataFrame. Table storing the regression results
    """
    results = None
    cls_desc = ["automl_model", "task_id", "time_limit", "accuracy",
                "balanced_accuracy", "F1_score", "AUC"]
    rgs_desc = ["automl_model", "task_id", "time_limit", "MSE", "MAE", "R2_score"]

    def __init__(self, supress_warnings=True, sess_name=""):
        if supress_warnings:
            warnings.filterwarnings('ignore')
        self.results = []
        self.sess_name = generate_rand_string(6) if not sess_name else sess_name
        self.cls_results = pd.DataFrame(columns=self.cls_desc)
        self.rgs_results = pd.DataFrame(columns=self.rgs_desc)

    def measure_performance_cls(self, y_true, y_pred, binary=False):
        """ Calculate the performance of the classification task
        # Arguments
            y_true: A numpy array containing the ground truth labels
            y_pred: A numpy array containing the predicted labels
            binary: Boolean specifying if the objective isbinary or multiclass
        # Returns
            list of the performance scores based on various evaluation metrics.
        """
        accuracy = accuracy_score(y_true, y_pred)
        ber = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="binary") if binary else f1_score(
            y_true, y_pred, average="weighted")
        auc = roc_auc_score(y_true, y_pred) if binary else "-"
        return [accuracy, ber, f1, auc]

    def measure_performance_rgs(self, y_true, y_pred):
        """ Calculate the performance of the regression task
        # Arguments
            y_true: A numpy array containing the ground truth
            y_pred: A numpy array containing the predicted values
        # Returns
            list of the performance scores based on various evaluation metrics.
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return [mse, mae, r2]

    def export_results(self):
        """ Writes the results to a CSV file.
        # Arguments
            None
        # Returns
            None
        """
        if len(self.cls_results) > 0:
            self.cls_results.to_csv(self.sess_name + "_classification_results.csv",
                                    index=False)
        if len(self.rgs_results) > 0:
            self.rgs_results.to_csv(self.sess_name + "_regression_results.csv",
                                    index=False)

    @abstractmethod
    def evaluate(self, task, time_limit):
        """ Evaluates the performance of the single task.
        # Arguments
            task: Id of the OpenML task flow
            time_limit: Budget for the given task
        # Returns
            List of performance scores of the autoML system on the given task.
        """
        pass

    def run_automation(self, task_list, time_limit=10 * 60):
        """ Evaluate the list of the tasks in sequence
        # Arguments
            task_list: List of OpenML task ids
            time_limit: Budget for each of the task
        # Returns
            None
        """
        for task in task_list:
            try:
                self.evaluate(task, time_limit=time_limit)
                self.export_results()
            except:
                print("task: {} didnt work".format(task))

    def time_lapse(self, task_id,
                   time_limits=[30, 40, 50, 60, 90, 120, 150, 180, 240, 300]):
        """ Evaluate the task on different time_limits
        # Arguments
            task_id: Id of the OpenML task flow
            time_limits: List of the time_limits to test the performance on
        # Returns
            List of combined results of the autoML on each of the time_limit
        This function evaluates and compares the performance of the autoML system
        on different time_limits. It is helpful to understand the amount of
        improvement with increase in time budget
        """
        tl_results = []
        for time_limit in time_limits:
            tl_results.append(self.evaluate(task_id, time_limit=time_limit))
        return tl_results

    def get_dataset_splits(self, task_id):
        """ Get the train/test splits for the given task
        # Arguments
            task_id: Id of OpenML task flow
        # Returns
            Train/Test datasets in numpy array format
        """
        task = openml.tasks.get_task(task_id)
        train_indices, test_indices = task.get_train_test_split_indices()
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=task.target_name, dataset_format='array')

        x_train, y_train = X[train_indices], y[train_indices]
        x_test, y_test = X[test_indices], y[test_indices]
        return x_train, y_train, x_test, y_test


class BenchmarkingAutoKaggle(BenchmarkingBase):
    """ Extends the benchmarking class for evaluating AutoKaggle.

        This class evaluates the performance of AutoKaggle on the input
        classification or regression task_list.
    """

    def get_data_info(self, dataset, num_cols):
        """ Get the info of each feature data type
        # Arguments
            dataset: dataset id in OpenML
            num_cols: Total number of columns
        # Returns
            A numpy array containing the data_type of each feature column
        """
        nominal_feat = dataset.get_features_by_type('nominal')
        numerical_feat = dataset.get_features_by_type('numeric')
        string_feat = dataset.get_features_by_type('string')
        date_feat = dataset.get_features_by_type('date')

        data_info = []
        for i in range(num_cols):
            if i in date_feat:
                data_info.append("TIM")
            elif i in numerical_feat:
                data_info.append("NUM")
            else:
                data_info.append("CAT")
        return np.array(data_info)

    def evaluate(self, task_id, time_limit=10 * 60):
        """
            See base class.
        """
        task_info = ["autokaggle", task_id, time_limit]
        task = openml.tasks.get_task(task_id)
        train_indices, test_indices = task.get_train_test_split_indices()
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=task.target_name, dataset_format='array')

        x_train, y_train = X[train_indices], y[train_indices]
        x_test, y_test = X[test_indices], y[test_indices]

        # Create feature type list from openml.org indicator
        data_info = self.get_data_info(dataset, len(attribute_names))

        # Train
        if task.task_type == 'Supervised Classification':
            automl = Classifier()
        elif task.task_type == 'Supervised Regression':
            automl = Regressor()
        else:
            print("UNSUPPORTED TASK_TYPE")
            assert (0)

        automl.fit(x_train, y_train, time_limit=time_limit, data_info=data_info)

        # Evaluate
        y_hat = automl.predict(x_test)

        if task.task_type == 'Supervised Classification':
            is_binary = True if len(task.class_labels) <= 2 else False
            result = task_info + self.measure_performance_cls(y_test, y_hat,
                                                              binary=is_binary)
            self.cls_results.loc[len(self.cls_results)] = result
        elif task.task_type == 'Supervised Regression':
            result = task_info + self.measure_performance_rgs(y_test, y_hat)
            self.rgs_results.loc[len(self.rgs_results)] = result
        print(result)
        return result

    #
    # class BenchmarkingAutoSklearn(BenchmarkingBase):
    """ Extends the benchmarking class for evaluating AutoSklearn.
    
        This class evaluates the performance of AutoKaggle on the input 
        classification or regression task_list.
    """


#     def get_data_info(self, categorical_indicator):
#         return ['Categorical' if ci else 'Numerical' for ci in categorical
#         indicator]
#
#     def evaluate(self, task_id, time_limit=10*60):
#         task_info = ["autosklearn", task_id, time_limit]
#         task = openml.tasks.get_task(task_id)
#         train_indices, test_indices = task.get_train_test_split_indices()
#         dataset = task.get_dataset()
#         X, y, categorical_indicator, attribute_names = dataset.get_data(
#         target=task.target_name, dataset_format='array')
#
#         x_train, y_train = X[train_indices], y[train_indices]
#         x_test, y_test = X[test_indices], y[test_indices]
#
#         # Create feature type list from openml.org indicator
#         feat_type = self.get_data_info(categorical_indicator)
#
#         # Train
#         if task.task_type == 'Supervised Classification':
#             automl = AutoSklearnClassifier(
#                 time_left_for_this_task=time_limit,
#                 per_run_time_limit=time_limit//10, **kwargs)
#         elif task.task_type == 'Supervised Regression':
#             automl = AutoSklearnRegressor(
#                 time_left_for_this_task=time_limit,
#                 per_run_time_limit=time_limit//10, **kwargs)
#         else:
#             print("UNSUPPORTED TASK_TYPE")
#             assert(0)
#
#         automl.fit(x_train, y_train, feat_type=feat_type)
#
#         y_hat = automl.predict(x_test)
#         if task.task_type == 'Supervised Classification':
#             is_binary = True if len(task.class_labels) <= 2 else False
#             result = task_info + self.measure_performance_cls(y_test, y_hat,
#             binary=is_binary)
#             self.cls_results.loc[len(self.cls_results)] = result
#         elif task.task_type == 'Supervised Regression':
#             result = task_info + self.measure_performance_rgs(y_test, y_hat)
#             self.rgs_results.loc[len(self.rgs_results)] = result
#         self.results.append(result)
#         print(result)
#         return result


def get_dataset_ids(task_ids):
    """ Fetches the dataset_ids.
    # Arguments
        task_ids: List of ids of OpenML task flows
    # Returns
        dataset_list: List of the dataset Ids
    """
    if type(task_ids) == list:
        return [openml.tasks.get_task(t_id).dataset_id for t_id in task_ids]
    else:
        return openml.tasks.get_task(task_ids).dataset_id


def get_task_info(task_ids):
    """ Fetches the dataset_ids and the task objective.
    # Arguments
        task_ids: List of ids of OpenML task flows.
    # Returns
        dataset_list: List of the dataset Ids.
        task_types: List of the task type (such as 'binary/multiclass
        classification' or 'regression'
    """
    task_types = []
    dataset_list = []
    for i, t_id in enumerate(task_ids):
        task = openml.tasks.get_task(t_id)
        dataset = openml.datasets.get_dataset(task.dataset_id)
        if task.task_type_id == 1:
            _, y, _, _ = dataset.get_data(target=task.target_name,
                                          dataset_format='array')
            task_type = "Binary Classification" if len(
                set(y)) <= 2 else "Multiclass classification ({})".format(
                len(set(y)))
        else:
            task_type = "Regression"
        task_types.append(task_type)
        dataset_list.append(dataset)
    return dataset_list, task_types


def get_dataset_properties(task_ids):
    """ Fetches the properties of the dataset for given task flow id
    # Arguments
        task_ids: List of ids of OpenML task flows
    # Returns
        Dataframe containing the info of each of the dataset.
    This function provides the dataset info such as number of instances, number of
    numeric/nominal/string columns etc.
    """
    dataset_list, task_types = get_task_info(task_ids)
    df = pd.DataFrame(
        columns=["Name", "#Samples", "Task_Type", "#Numeric", "#Nominal", "#String",
                 "#Date"])
    for i, dataset in enumerate(dataset_list):
        df.loc[i] = [
            dataset.name,
            dataset.qualities["NumberOfInstances"],
            task_types[i],
            len(dataset.get_features_by_type('numeric')),
            len(dataset.get_features_by_type('nominal')),
            len(dataset.get_features_by_type('string')),
            len(dataset.get_features_by_type('date')),
        ]
    return df


def get_performance_table(filename, metric):
    """ Generates a comprehensive report table of AutoML performance.
    # Arguments
        filename: A csv file containing the results of AutoML runs
        metric: Scoring metric to be used for comparison
    # Returns
        Pandas Dataframe listing the performance of different AutoML systems on
        the given datasets.
    This function reads the results csv and converts it into the performance table
    based on the median of the results for each task.
    """
    test = pd.read_csv(filename)
    perf = pd.DataFrame(columns=["Name", "AutoKaggle", "AutoSklearn", "H2O.ai"])
    task_ids = list(set(test["task_id"]))
    dataset_ids = get_dataset_ids(task_ids)

    test = test.set_index(["task_id", "automl_model"])
    test.sort_index(inplace=True)
    for i, t_id in enumerate(task_ids):
        try:
            name = openml.datasets.get_dataset(dataset_ids[i]).name
            auto_kaggle = test.loc[(t_id, "autokaggle")][metric].median()\
                if (t_id, "autokaggle") in test.index else np.nan
            auto_sklearn = test.loc[(t_id, "autosklearn")][metric].median()\
                if (t_id, "autosklearn") in test.index else np.nan
            h2o_ai = test.loc[(t_id, "autosklearn")][metric].median()\
                if (t_id, "autosklearn") in test.index else np.nan
            perf.loc[i] = [name, auto_kaggle, auto_sklearn, h2o_ai]
        except Exception as e:
            print(e)
    return perf


def style_results(res):
    """ Highlights the best result in the results column
    # Arguments
        res: Dataframe containing the results of various AutoML runs
    # Returns
        Highlighed data-frame
    """

    def highlight_max(s):
        """
        Highlight the maximum in a Series yellow.
        """
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    res = res.set_index("Name")
    res.style.apply(highlight_max, axis=1)
    return res


def get_box_plot(results, task_id, metric):
    """ Generates a box plot of the variance in the result.
    # Arguments
        results: Results of various runs using AutoML systems
        task_id: Id for OpenML task flow
        metric: Score metric considered for the box-plot
    # Returns
        None
    Builds and displays the box plot showing the variance in results for the
    AutoML performance on the given dataset.
    """
    auto_sklearn = list(results.loc[(task_id, "autosklearn")][metric])
    auto_kaggle = list(results.loc[(task_id, "autokaggle")][metric])
    med_sk = statistics.median(auto_sklearn)
    med_ak = statistics.median(auto_kaggle)
    while len(auto_sklearn) < len(auto_kaggle):
        auto_sklearn.append(med_sk)
    while len(auto_sklearn) > len(auto_kaggle):
        auto_kaggle.append(med_ak)
    temp = pd.DataFrame(
        data={"Autokaggle": auto_kaggle, "AutoSklearn": auto_sklearn})
    temp.boxplot()


if __name__ == "__main__":
    regression_task_list = [52948, 2295, 4823, 2285, 4729, 4990, 4958, 2280, 4834,
                            4850, 4839]
    classification_task_list = [3021, 45, 2071, 2076, 3638, 3780, 3902, 3945, 3954,
                                14951, 59, 24, 146230, 31, 10101,
                                9914, 3020, 3524, 3573, 3962]
    ak = BenchmarkingAutoKaggle(sess_name='test_perf')
    import time

    # t1 = time.time()
    # for _ in range(1):
    #     ak.run_automation(classification_task_list)
    # t2 = time.time()
    # print(t2-t1)
    np.random.seed(1001)
    random.seed(1001)
    import time

    t1 = time.time()
    ak.evaluate(3021)
    t2 = time.time()
    print(t2 - t1)
