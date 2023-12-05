# NewYorkAirBnBRegressionAnalysis


Real estate price prediction in Washington state

Team member list:
Rahul Galgali
Aishwariya Chunduru
Prathamesh Jalgaonkar
Himali Shewale
Vaishnavi Inamdar
Name: Real estate price prediction in Washington state.
Description: 
This project was created by our group for MIS 691: Decision Support Systems. 
The primary goal of our House Price Prediction Regression Project is to develop a robust and accurate model that can estimate property prices based on various features such as location, square footage, number of bedrooms and bathrooms, proximity to amenities, and other relevant factors. By harnessing the predictive capabilities of regression analysis, we aim to empower homeowners, real estate agents, and investors with valuable insights into potential property values.

Dependencies:
!pip install pandas numpy scikit-learn matplotlib kaggle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


Kaggle API setup:wsx 
os.environ['KAGGLE_USERNAME'] = 'aishwariyachunduru' 
os.environ['KAGGLE_KEY'] = '893ccea6d7d2d59961d5feaffe4afe1d' 
dataset_url = "https://www.kaggle.com/shree1992/housedata/download"
Data loading:
!kaggle datasets download -d $dataset_url !unzip housedata.zip
house_data = pd.read_csv("data.csv", error_bad_lines=False)
Number of records in the dataset
n = len(house_data) 
print(n)
Pre-processing steps - Handling missing values
print(sum(house_data['price'] == 0))
# Count the number of zero values in the 'price' column
house_data['price'].replace(0, np.nan, inplace=True)
# Replace all zero values with NaN
print(sum(house_data['price'].isna()))
# Count the number of NaN values in the 'price' column
house_data['price'].fillna(house_data['price'].mean(), inplace=True)
# Replace all NaN values with the mean of the 'price' column


Data Exploration - Variation of prices over the years
avg_prices = house_data.groupby('yr_built')['price'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x='yr_built', y='price', data=avg_prices)
plt.xlabel('Year Built')
plt.ylabel('Price')
plt.title('Prices of houses over the years')
plt.show()

City
# City
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='city', hue='city', data=house_data, alpha=0.5)
plt.xlabel('Price')
plt.ylabel('City')
plt.title('House prices by City')
plt.show()

Pie chart for 'view'
view_counts = house_data['view'].value_counts()
plt.figure(figsize=(10, 6))
plt.pie(view_counts, labels=view_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of View')
plt.show()

Pie chart for 'condition'
condition_counts = house_data['condition'].value_counts()
plt.figure(figsize=(10, 6))
plt.pie(condition_counts, labels=condition_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Condition')
plt.show()

Pie chart for 'waterfront'
waterfront_counts = house_data['waterfront'].value_counts()
plt.figure(figsize=(10, 6))
plt.pie(waterfront_counts, labels=waterfront_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Waterfront')
plt.show()

Histogram
quant_house_data = house_data[['price', 'bedrooms', 'sqft_living', 'floors', 'sqft_lot', 'sqft_above', 'sqft_basement', 'yr_renovated', 'yr_built']]
quant_house_data.hist(bins=20, figsize=(15, 10))
plt.show()

Checking for missing values in the 'price' variable
print(sum(quant_house_data['price'].isna()))
Creating a new dataset without missing values in the 'price' variable
quant_house_data_clean = quant_house_data.dropna()
Creating a histogram of the 'price' variable without missing values
plt.figure(figsize=(10, 6))
sns.histplot(quant_house_data_clean['price'], bins=20, kde=True)
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Histogram of Price')
plt.show()

Correlation matrix
correlation_matrix = house_data[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

Feature transformation - Transforming sqft basement - 1 if basement is there, 0 if basement is not there
house_data['sqft_basement'] = np.where(house_data['sqft_basement'] > 0, 1, 0)
No missing data
print(house_data.isna().sum())
Displaying details of the columns
print(house_data.info())
Dropping date, street, city, statezip, country columns
house_data.drop(['date', 'street', 'city', 'statezip', 'country'], axis=1, inplace=True)
Forward stepwise selection - to determine the predictors
X = house_data.drop('price', axis=1) 
y = house_data['price']

Model training and evaluation:
Linear Regression
lm = LinearRegression()
lm.fit(X, y)
lm_pred = lm.predict(X)
mse_linear_reg = mean_squared_error(y, lm_pred)
rmse_linear_reg = np.sqrt(mse_linear_reg)
print(f"RMSE of Linear Regression: {rmse_linear_reg}")

Ridge Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
ridge = RidgeCV(alphas=np.logspace(-6, 6, 13))
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
mse_ridge = mean_squared_error(y_test, ridge_pred)
rmse_ridge = np.sqrt(mse_ridge)
print(f"RMSE of Ridge Regression: {rmse_ridge}")

Lasso Regression
lasso = LassoCV(alphas=np.logspace(-6, 6, 13), max_iter=1000)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
mse_lasso = mean_squared_error(y_test, lasso_pred)
rmse_lasso = np.sqrt(mse_lasso)
print(f"RMSE of Lasso Regression: {rmse_lasso}")

Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, rf_pred)
rmse_rf = np.sqrt(mse_rf)
print(f"RMSE of Random Forest: {rmse_rf}")

Results of all models:

RMSE of Linear Regression: 492655.9754361901
RMSE of Ridge Regression: 749596.1769284814
RMSE of Lasso Regression: 749626.2519488161
RMSE of Random Forest: 768296.9049585541


Software used:
1. Python: The primary programming language for the project.
2. Jupyter Notebooks: Commonly used for interactive data exploration and analysis.
3. Pandas: A powerful data manipulation library in Python.
4. NumPy: Used for numerical operations and array manipulations.
5. scikit-learn: A machine learning library for building and evaluating models.
6. Matplotlib: A plotting library for creating visualizations.
7. Seaborn: A statistical data visualization library based on Matplotlib.
8. Kaggle API: Utilized for downloading datasets from Kaggle.

Licenses:
1. Python License (PSF): Python Software Foundation License is an open-source license.
2. Jupyter License: Jupyter is open-source and primarily under the modified BSD license.
3. Pandas License (BSD): pandas is released under the 3-Clause BSD license.
4. NumPy License (BSD): Similar to pandas, NumPy also uses the 3-Clause BSD license.
5. scikit-learn License (BSD): scikit-learn follows the permissive 3-Clause BSD license.
6. Matplotlib License (Matplotlib License): Matplotlib uses a custom license similar to the Python Software Foundation License.
7. Seaborn License (BSD): Seaborn is released under the 3-Clause BSD license.
8. Kaggle API License (Apache 2.0): Kaggle API follows the Apache 2.0 License.

Conclusion:

In summarising our Real Estate Price Prediction project, we aimed to estimate property prices in Washington state, unravelling the various factors influencing these valuations. We embarked on a journey of data preprocessing, exploration, and model training, unearthing intriguing patterns such as the fluctuation of prices over the years and the distinct impact of different cities on housing costs. The evaluation of Linear Regression, Ridge Regression, Lasso Regression, and Random Forest models revealed that Linear Regression exhibited superior performance, as reflected in the lower RMSE.

For stakeholders navigating the complex landscape of Washington State's real estate, our project equips them with valuable tools for making informed decisions. As we look forward, refining the models and incorporating additional features holds the promise of enhancing prediction accuracy, paving the way for future applications and deeper insights in the field of real estate data analysis.

Contact:
For further queries please contact Rahul Galgali at rgalgali1050@sdsu.edu. 
