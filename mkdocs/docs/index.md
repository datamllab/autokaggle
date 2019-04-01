
<div style="text-align: center">
<p>
<a class="badge-align" href="https://travis-ci.org/datamllab/autokaggle"><img alt="Build Status" src="https://travis-ci.org/datamllab/autokaggle.svg?branch=master"/></a>
<a class="badge-align" href="https://badge.fury.io/py/autokaggle"><img src="https://badge.fury.io/py/autokaggle.svg" alt="PyPI version"></a>
</p>
</div>

Auto-Kaggle is an open source software library for automated kaggle competition.
It is developed by <a href="http://faculty.cs.tamu.edu/xiahu/index.html" target="_blank" rel="nofollow">DATA Lab</a> at Texas A&M University and community contributors.

## Installation


To install the package, please use the `pip` installation as follows:

    pip install autokaggle
    
**Note:** currently, Auto-Kaggle is only compatible with: **Python 3.6**.

## Example

Here is a short example of using the package.


    import autokaggle as ak

    clf = ak.TabularClassifier(verbose=True)
    clf.fit(x_train, y_train, time_limit=12 * 60 * 60, data_info=datainfo)


## Community

 
## Support Auto-Kaggle

We accept donations on [Open Collective](https://opencollective.com/autokaggle).
Thank every backer for supporting us!


## DISCLAIMER

Please note that this is a **pre-release** version of the Auto-Kaggle which is still undergoing final testing before its official release. The website, its software and all content found on it are provided on an
“as is” and “as available” basis. Auto-Kaggle does **not** give any warranties, whether express or implied, as to the suitability or usability of the website, its software or any of its content. Auto-Kaggle will **not** be liable for any loss, whether such loss is direct, indirect, special or consequential, suffered by any party as a result of their use of the libraries or content. Any usage of the libraries is done at the user’s own risk and the user will be solely responsible for any damage to any computer system or loss of data that results from such activities. Should you encounter any bugs, glitches, lack of functionality or
other problems on the website, please let us know immediately so we
can rectify these accordingly. Your help in this regard is greatly
appreciated.

## Acknowledgements

The authors gratefully acknowledge the D3M program of the Defense Advanced Research Projects Agency (DARPA) administered through AFRL contract FA8750-17-2-0116; the Texas A&M College of Engineering, and Texas A&M. 
