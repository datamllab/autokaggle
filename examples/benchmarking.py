import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.datasets
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score,\
mean_absolute_error, mean_squared_error
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.classification import AutoSklearnClassifier
from autokaggle import *
from autokaggle.utils import *
import openml
openml.config.apikey = '3c7196c92a274c3b9405a7e26e9f848e'
import warnings
from abc import abstractmethod

def generate_rand_string(size):
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))

class BenchmarkingBase:
    results = None
    cls_desc = ["automl_model", "task_id", "time_limit", "accuracy", "balanced_accuracy", "F1_score", "AUC"]
    rgs_desc = ["automl_model", "task_id", "time_limit", "MSE", "MAE", "R2_score"]
    
    def __init__(self, supress_warnings=True, sess_name=""):
        if supress_warnings:
            warnings.filterwarnings('ignore')
        self.results = []
        if not sess_name:
            sess_name = generate_rand_string(6)
        self.cls_results = pd.DataFrame(columns=self.cls_desc)
        self.rgs_results = pd.DataFrame(columns=self.rgs_desc)
        
    def measure_performance_cls(self, y_true, y_pred, binary=False):
        accuracy = accuracy_score(y_true, y_pred)
        ber = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="binary") if binary else f1_score(y_true, y_pred, average="weighted")
        auc = roc_auc_score(y_true, y_pred) if binary else "-"
        return [accuracy, ber, f1, auc]

    def measure_performance_rgs(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return [mse, mae, r2]
    
    def export_results(self):
        self.cls_results.to_csv(self.sess_name + "_classification_results.csv", index=False)
        self.rgs_results.to_csv(self.sess_name + "_regression_results.csv", index=False)
    
    @abstractmethod
    def evaluate(self, task, time_limit):
        pass
        
    def run_automation(self, task_list, time_limit=10*60):
        for task in task_list:
            try:
                self.evaluate(task, time_limit=time_limit)
            except:
                print("task: {} didnt work".format(task))
                
    def time_lapse(self, task_id, time_limits=[30, 40, 50, 60, 90, 120, 150, 180, 240, 300]):
        tl_results = []
        for time_limit in time_limits:
            tl_results.append(self.evaluate(task_id, time_limit=time_limit))
        return tl_results
    
    
class BenchmarkingAutoKaggle(BenchmarkingBase):
    def get_data_info(self, dataset, num_cols):
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
    
    def evaluate(self, task_id, time_limit=10*60):
        task_info = ["autokaggle", task_id, time_limit]
        task = openml.tasks.get_task(task_id)
        train_indices, test_indices = task.get_train_test_split_indices()
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(target=task.target_name, dataset_format='array')

        x_train, y_train = X[train_indices], y[train_indices]
        x_test, y_test = X[test_indices], y[test_indices]

        # Create feature type list from openml.org indicator
        data_info = self.get_data_info(dataset, len(attribute_names))

        # Train
        if task.task_type == 'Supervised Classification':
            automl = AutoKaggle()
        elif task.task_type == 'Supervised Regression':
            automl = AutoKaggle(LgbmRegressor)
        else:
            print("UNSUPPORTED TASK_TYPE")
            assert(0)

        automl.fit(x_train, y_train, time_limit=time_limit, data_info=data_info)

        # Evaluate
        y_hat = automl.predict(x_test)
        
        if task.task_type == 'Supervised Classification':
            is_binary = True if len(task.class_labels) <= 2 else False
            result = task_info + self.measure_performance_cls(y_test, y_hat, binary=is_binary)
            self.cls_results.loc[len(self.cls_results)] = result
        elif task.task_type == 'Supervised Regression':
            result = task_info + self.measure_performance_rgs(y_test, y_hat)
            self.rgs_results.loc[len(sel.rgs_results)] = result
        print(result)
        return result

    
class BenchmarkingAutoSklearn(BenchmarkingBase):
    def get_data_info(self, categorical_indicator):
        return ['Categorical' if ci else 'Numerical' for ci in categorical_indicator]
    
    def evaluate(self, task_id, time_limit=10*60):
        task_info = ["autosklearn", task_id, time_limit]
        task = openml.tasks.get_task(task_id)
        train_indices, test_indices = task.get_train_test_split_indices()
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(target=task.target_name, dataset_format='array')

        x_train, y_train = X[train_indices], y[train_indices]
        x_test, y_test = X[test_indices], y[test_indices]

        # Create feature type list from openml.org indicator
        feat_type = self.get_data_info(categorical_indicator)

        # Train
        if task.task_type == 'Supervised Classification':
            automl = AutoSklearnClassifier(
                time_left_for_this_task=time_limit,
                per_run_time_limit=time_limit//10, **kwargs)
        elif task.task_type == 'Supervised Regression':
            automl = AutoSklearnRegressor(
                time_left_for_this_task=time_limit,
                per_run_time_limit=time_limit//10, **kwargs)
        else:
            print("UNSUPPORTED TASK_TYPE")
            assert(0)

        automl.fit(x_train, y_train, feat_type=feat_type)

        y_hat = automl.predict(x_test)
        if task.task_type == 'Supervised Classification':
            is_binary = True if len(task.class_labels) <= 2 else False
            result = task_info + self.measure_performance_cls(y_test, y_hat, binary=is_binary)
            self.cls_results.loc[len(self.cls_results)] = result
        elif task.task_type == 'Supervised Regression':
            result = task_info + self.measure_performance_rgs(y_test, y_hat)
            self.rgs_results.loc[len(self.rgs_results)] = result
        self.results.append(result)
        print(result)
        return result