# autokaggle
Automated Machine Learning (AutoML) for Kaggle Competition

### Automated tabular classifier tutorial.

Class `TabularClassifier` and `TabularRegressor` are designed for automated generate best performance shallow/deep architecture
for a given tabular dataset. (Currently, theis module only supports lightgbm classifier and regressor.)


```python
    clf = TabularClassifier(verbose=True)
    clf.fit(x_train, y_train, time_limit=12 * 60 * 60, data_info=datainfo)
```

* x_train: string format text data
* y_train: int format text label
* data_info: a numpy.array describing the feature types (time, numerical or categorical) of each column in x_train.


**Notes:** Preprocessing of the tabular data:
* Class `[TabularPreprocessor]` involves several automated feature preprocessing and engineering operation for tabular data . 
*The input data should be in numpy array format for the class `TabularClassifier` and `TabularRegressor` .
