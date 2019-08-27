# Cars_Regression_Python

### Project Objective
The goal of this project is to run some simple linear regressions on the [Auto-MPG dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data)
from the UCI Machine Learning Repository. This will also use techniques like OHE (one hot encoding), Train/Test datasets splits, setting the random seed
for reproducibility, and missing value imputation. This will likely act as a base script for much of the future work I will likely do in Python.

### Required Packages
* numpy
* pandas
* sklearn
* statsmodels.api
* urllib.request

### Notes
- csv.reader seems needlessly complex since I have to initialize the fields and rows, np.genfromtext is much simpler
- Train/Test dataset splits utilize a randomized (reproducible) index
- I should probably bundle my imputation scripts into a module that I can call similar to how I handled DStools/DataframeCompare for R
- statsmodels is seems very similar to lm (R) and can produce a regression summary whereas sklearn.linear_model needs each component to be called
- Probably worth exploring how penalized regression and cross-validation techniques can be further utilized
