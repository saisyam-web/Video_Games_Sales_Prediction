# Importing the required libraries
import pandas as pd
import numpy as np

# Importing the dataset
dataset = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')

# Dropping certain less important features
dataset.drop(columns = ['Year_of_Release', 'Developer', 'Publisher', 'Platform'], inplace = True)

# To view the columns with missing values
print('Feature name || Total missing values')
print(dataset.isna().sum())

X = dataset.iloc[:, :].values
X = np.delete(X, 6, 1)

y = dataset.iloc[:, 6:7].values

# Splitting the dataset into Train and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Saving name of the games in training and test set
games_in_training_set = X_train[:, 0]
games_in_test_set = X_test[:, 0]

# Dropping the column that contains the name of the games
X_train = X_train[:, 1:]
X_test = X_test[:, 1:]

from sklearn.impute import SimpleImputer
simpleImputer = SimpleImputer(strategy = 'mean')
X_train[:, [5 ,6, 7, 8]] = simpleImputer.fit_transform(X_train[:, [5, 6, 7, 8]])
X_test[:, [5 ,6, 7, 8]] = simpleImputer.transform(X_test[:, [5, 6, 7, 8]])

#from sklearn_pandas import CategoricalImputer
from feature_engine.imputation import CategoricalImputer
categorical_imputer = CategoricalImputer(imputation_method = 'missing', fill_value = 'NA')
X_train[:, [0, 9]] = categorical_imputer.fit_transform(X_train[:, [0, 9]])
X_test[:, [0, 9]] = categorical_imputer.transform(X_test[:, [0, 9]])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0, 9])], remainder = 'passthrough') 
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

from xgboost import XGBRegressor
model = XGBRegressor(n_estimators = 200, learning_rate= 0.08)
model.fit(X_train, y_train)

# Predicting test set results
y_pred = model.predict(X_test)

# Visualising actual and predicted sales
games_in_test_set = games_in_test_set.reshape(-1, 1)
y_pred = y_pred.reshape(-1, 1)
predictions = np.concatenate([games_in_test_set, y_pred, y_test], axis = 1)
predictions = pd.DataFrame(predictions, columns = ['Name', 'Predicted_Global_Sales', 'Actual_Global_Sales'])
print(predictions)
