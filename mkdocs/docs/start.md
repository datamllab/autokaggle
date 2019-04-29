# Getting Started

---

## Installation
The installation of Auto-Kaggle is the same as other python packages. 

**Note:** currently, Auto-Kaggle is only compatible with: **Python 3.6**.

### Latest Stable Version (`pip` installation):
You can run the following `pip` installation command in your terminal to install the latest stable version.

    pip install autokaggle

### Bleeding Edge Version (manual installation):
If you want to install the latest development version. 
You need to download the code from the GitHub repo and run the following commands in the project directory.

    pip install -r requirements.txt
    python setup.py install




## A Simple Example

We show an example of binary classification using the randomly generated data.



### Data with numpy array (.npy) format.
[[source]](https://github.com/datamllab/autokaggle/blob/master/examples/tabular_classification_binary.py)


If the train/test data are already formatted into numpy arrays, you can 

    import numpy as np
    from autokeras import TabularClassifier

    if __name__ == '__main__':
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
    
        clf = TabularClassifier()
        datainfo = np.array(['TIME'] * ntime + ['NUM'] * nnum + ['CAT'] * ncat)
        clf.fit(x_train, y_train, time_limit=12 * 60 * 60, data_info=datainfo)
    
        AUC = clf.evaluate(x_test, y_test)
        print(AUC)
    
        
In the example above, the train/test data are already formatted into numpy arrays.



