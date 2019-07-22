##########################################################
#
#   Simple Linear Regression Example
#
#   Objective: Import Cars data file from .csv file
#       downloaded from UCI Machine Learning Repository.
#       Create a simple linear regression to solve for
#       MPG. Use this script as a base script for future
#       work in Python.
#
#   Initial Build: 4/7/2018
#
#   Change Log:
#   7/9/2019
#       - Switch import dependency away from CSV to
#           urllib2
#########################################################

# Import required modules
import numpy as np  # Import NumPy
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
# import csv
from urllib.request import urlopen

# Define Random_Seed
Random_Seed = 10

# Import .csv file using NumPy (PREFERRED)
# Data = np.genfromtxt(
#        'C:/Users/Chris Castillo/Data Science/Projects/Cars - Linear Regression/20180407_Cars Data.csv'
#        , dtype = None
#        , delimiter = ','
#        , names = True
#        , skip_header = 0
#        )

# Create Data object using UCI Machine Learning Repository link
Data = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    , sep="\s+"
    , header=None
    , names=[
        'mpg'
        , 'cylinders'
        , 'displacement'
        , 'horsepower'
        , 'weight'
        , 'acceleration'
        , 'model_year'
        , 'origin'
        , 'car_name'
    ]
    , verbose=True
    )

# # Define csv filepath
# filepath = 'C:/Users/Chris Castillo/Data Science/Projects/Cars - Linear Regression/20180407_Cars Data.csv'
#
# # Initialize the titles and rows list
# fields = []
# rows = []
#
# # Read the csv file
# with open(filepath, 'r') as csvfile:
#     # Create a csv reader object
#     Data = csv.reader(
#         csvfile
#         , delimiter=','
#     )
#
#     # Get field names from first row of csv
#     fields = next(Data)
#
#     # Extract each data row one by one and append into rows list
#     for row in Data:
#         rows.append(row)
#
#     # Print total number of rows
#     print("Total number of rows: %d" % Data.line_num)

# # Create DataFrame from csv data
# Data = pd.DataFrame(rows
#                     , columns=fields
#                     )
#
# # Fix "mpg" column within DataFrame
# Data.rename(columns={"ï»¿mpg": "mpg"
#     , "model year": "model_year"
#     , "car name": "car_name"
#                      }, inplace=True)

# See top 5 rows of Data
Data.head()

# Get the data types for Data (DataFrame)
Data.dtypes

# Get summary statistics on Data, must use argument for DataFrame with mixed data types
Data.describe(include='all')

# Get dimensions of Data
Data.shape

# List the unique values for 'origin', get the frequency of the unique values using groupby functionality
Data['origin'].unique()
Data.groupby('origin')['origin'].count()

# Fix the dtypes for fields within Data (DataFrame)
Data['mpg'] = Data['mpg'].astype('float')
Data['cylinders'] = Data['cylinders'].astype('int')
Data['displacement'] = Data['displacement'].astype('float')
Data['horsepower'] = pd.to_numeric(Data['horsepower'], errors='coerce')
Data['weight'] = Data['weight'].astype('int')
Data['acceleration'] = Data['acceleration'].astype('float')
Data['model_year'] = Data['model_year'].astype('int')
Data['origin'] = Data['origin'].astype('object')

# Create another feature for horsepower/weight
Data['wgt_hp_ratio'] = pd.Series(Data['horsepower'] / Data['weight'])

# Create lists of Numeric and Factor fields within Data
Data_Numeric_Fields = list(Data.select_dtypes(include=[np.number]).columns.values)
Data_Object_Fields = list(Data.select_dtypes(include=[np.object]).columns.values)

# Generate a random sample of integers without replacement using 50% of the Data.index
np.random.seed(Random_Seed)
Random_Rows = np.random.choice(
    a=len(Data.index)
    , size=int(round(len(Data.index) * 0.50, 0))
    , replace=False
)

# Separate Data into Train_Data and Test_Data (~ means not)
Train_Data = Data.loc[Data.index.isin(Random_Rows)]
Test_Data = Data.loc[~Data.index.isin(Random_Rows)]

# *****************************************
# *****************************************
# **** Missing Train_Data Imputation ******
# *****************************************
# *****************************************

# Identify Numeric & Object columns with missing values
Train_Data_Missing_Numeric = Train_Data[Data_Numeric_Fields].columns[
    Train_Data[Data_Numeric_Fields].isna().any()].tolist()
Train_Data_Missing_Object = Train_Data[Data_Object_Fields].columns[Train_Data[Data_Object_Fields].isna().any()].tolist()

# Temporarily turn off warnings about rewriting data in original DataFrame
pd.options.mode.chained_assignment = None  # default='warn'

# Impute missing Numeric values in Train_Data with median column value
if len(Train_Data_Missing_Numeric) > 0:
    for i in Train_Data_Missing_Numeric:
        print("Imputing NaN values for " + i + " using median value")

        Train_Data[i][pd.isnull(Train_Data[i])] = np.nanmedian(Train_Data[i])

        # End FOR LOOP

else:
    print("No Numeric columns require imputation")
# End IF Check


# Impute missing Object values in Train_Data with randomly sampled column value
if len(Train_Data_Missing_Object) > 0:
    for i in Train_Data_Missing_Object:
        print("Imputing NaN values for " + i + " using randomly sampled values")

        # Count how many records require imputation
        z = sum(pd.isnull(Train_Data[i]))

        # Create vector of values after excluding NaN
        No_Missing_Vector = Train_Data[i][pd.notnull(Train_Data[i])]

        # Set randomization seed and replace NaN values with a random sample (with replacement)
        np.random.seed(Random_Seed)
        Train_Data[i][pd.isnull(Train_Data[i])] = np.random.choice(a=No_Missing_Vector, size=z, replace=True)

        del z, No_Missing_Vector
        # End FOR LOOP

else:
    print("No Object columns require imputation")
# End IF Check

# Turn back on warnings about rewriting data in original DataFrame
pd.options.mode.chained_assignment = 'warn'

# *****************************************
# *****************************************
# ***** Missing Test_Data Imputation ******
# *****************************************
# *****************************************

# Identify Numeric & Object columns with missing values
Test_Data_Missing_Numeric = Test_Data[Data_Numeric_Fields].columns[Test_Data[Data_Numeric_Fields].isna().any()].tolist()
Test_Data_Missing_Object = Test_Data[Data_Object_Fields].columns[Test_Data[Data_Object_Fields].isna().any()].tolist()

# Temporarily turn off warnings about rewriting data in original DataFrame
pd.options.mode.chained_assignment = None  # default='warn'

# Impute missing Numeric values in Test_Data with median column value from Train_Data
if len(Test_Data_Missing_Numeric) > 0:
    for i in Test_Data_Missing_Numeric:
        print("Imputing NaN values for " + i + " using median value")

        Test_Data[i][pd.isnull(Test_Data[i])] = np.nanmedian(Train_Data[i])

        # End FOR LOOP

else:
    print("No Numeric columns require imputation")
# End IF Check


# Impute missing Object values in Test_Data with randomly sampled column value from Train_Data
if len(Test_Data_Missing_Object) > 0:
    for i in Test_Data_Missing_Object:
        print("Imputing NaN values for " + i + " using randomly sampled values")

        # Count how many records require imputation
        z = sum(pd.isnull(Test_Data[i]))

        # Create vector of values after excluding NaN
        No_Missing_Vector = Train_Data[i][pd.notnull(Train_Data[i])]

        # Set randomization seed and replace NaN values with a random sample (with replacement)
        np.random.seed(Random_Seed)
        Test_Data[i][pd.isnull(Test_Data[i])] = np.random.choice(a=No_Missing_Vector, size=z, replace=True)

        del z, No_Missing_Vector
        # End FOR LOOP

else:
    print("No Object columns require imputation")
# End IF Check

# Turn back on warnings about rewriting data in original DataFrame
pd.options.mode.chained_assignment = 'warn'

# **************************************************************
# **************************************************************
# **** Separate Train/Test Data & Establish Target/Feature *****
# **************************************************************
# **************************************************************

# Create a dictionary with feature names and object with target variable
Model_Features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']
Model_Target = 'mpg'

# Create Train_Data/Test_Data feature matrices
Train_Data_X = Train_Data[Model_Features]
Test_Data_X = Test_Data[Model_Features]

# Create Train_Data/Test_Data target vectors
Train_Data_Y = Train_Data[Model_Target]
Test_Data_Y = Test_Data[Model_Target]

# Create dummy variable for Train_Data_X
for column in Train_Data_X.columns:
    if Train_Data_X[column].dtype == object:
        dummyCols = pd.get_dummies(data=Train_Data_X[column]
                                   , prefix=column
                                   , prefix_sep='_'
                                   , drop_first=True
                                   )
        Train_Data_X = Train_Data_X.join(dummyCols)
del Train_Data_X[column]


# ********************************
# ********************************
# **** Linear Model Training *****
# ********************************
# ********************************

# Create linear regression object
Train_regr = linear_model.LinearRegression()

# Train the model using Train_Data to target 'mpg'
Train_regr.fit(X=Train_Data_X, y=Train_Data_Y)

# Print model Feature coefficients
print('Coefficients: \n', Train_regr.coef_)

# Print model Intercept coefficients
print('Intercept: \n', Train_regr.intercept_)

# Print R^2 of model
print('R^2: \n', Train_regr.score(X=Train_Data_X, y=Train_Data_Y))

# Create DataFrame of feature names and coefficients
Regr_Results = pd.DataFrame(data={'Features': Train_Data_X.columns.values, 'Coefficients': Train_regr.coef_})

# **********************************************
# **********************************************
# **** Linear Model Training (statsmodels) *****
# **********************************************
# **********************************************

# Fit a linear model using statsmodels.OLS
Train_regr_stats = sm.OLS(endog=Train_Data_Y
                          , exog=sm.add_constant(Train_Data_X)
                          , missing='raise'
                          ).fit()

# Display summary statistics results from regression run
print(Train_regr_stats.summary())

# *********************************************
# *********************************************
# **** Linear Model Predict (statsmodels) *****
# *********************************************
# *********************************************

# Create dummy variable for Test_Data_X
for column in Test_Data_X.columns:
    if Test_Data_X[column].dtype == object:
        dummyCols = pd.get_dummies(data=Test_Data_X[column]
                                   , prefix=column
                                   , prefix_sep='_'
                                   , drop_first=True
                                   )
        Test_Data_X = Test_Data_X.join(dummyCols)
        del Test_Data_X[column]

# Create predictions on Test_Data_X
Test_regr_stats_predY = Train_regr_stats.predict(sm.add_constant(Test_Data_X))

# Create RMSE for validation set
Test_RMSE = np.sqrt(np.mean((Test_Data_Y - Test_regr_stats_predY) ** 2))
