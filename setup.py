from distutils.core import setup
from setuptools import find_packages

setup(
    name='autokaggle',
    packages=find_packages(exclude=('tests',)),
    install_requires=[
        'scipy==1.2.0',
        'numpy==1.22.0',
        'scikit-learn==0.20.2',
        'lightgbm==2.2.3',
        'pandas==0.24.1'
    ],
    version='0.1.0',
    description='AutoML for Kaggle Competitions',
    author='DATA Lab at Texas A&M University',
    author_email='jhfjhfj1@gmail.com',
    keywords=['AutoML', 'Kaggle'],
    classifiers=[]
)
