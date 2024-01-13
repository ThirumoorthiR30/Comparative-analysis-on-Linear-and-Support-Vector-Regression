# Comparative-analysis-on-Linear-and-Support-Vector-Regression
Here We have taken a data set with Various Countries and their AQI ( Air quality index ) values. This project is a machine learning regression task focused on predicting the 'PM2.5 AQI Value' based on the 'Ozone AQI Value.'
Air Pollution Prediction Readme
Overview
This repository contains a Python script for predicting PM2.5 Air Quality Index (AQI) values based on Ozone AQI values using Linear Regression and Support Vector Regression (SVR). The code involves data preparation, cleansing, manual data splitting, model training, and evaluation.

Requirements
Make sure you have the following Python libraries installed:

pandas
numpy
matplotlib
scikit-learn
You can install them using the following command:

bash
Copy code
pip install pandas numpy matplotlib scikit-learn
Usage
Clone the repository:

bash
Copy code
git clone https://github.com/your_username/air-pollution-prediction.git
Navigate to the project directory:

bash
Copy code
cd air-pollution-prediction
Replace the placeholder 'global air pollution dataset.csv' with your actual dataset file in CSV format.

Run the Python script:

bash
Copy code
python air_pollution_prediction.py
Data Preparation
The dataset is loaded from a CSV file using pandas. Rows with NaN values in the target variable 'PM2.5 AQI Value' are dropped to ensure data quality.

Model Training
The script uses Linear Regression and Support Vector Regression (SVR) for predicting PM2.5 AQI values. The data is manually split into training and testing sets, and models are trained accordingly.

Linear Regression without Fit Method
Linear Regression parameters (slope and intercept) are calculated manually without using the fit method. Predictions are made on the test set using the derived parameters.

Support Vector Regression
Support Vector Regression is implemented using the SVR model from scikit-learn with a radial basis function (RBF) kernel. Predictions are made on the test set using the trained SVR model.

Evaluation Metrics
The performance of both models is evaluated using Mean Absolute Error (MAE) and R-squared coefficient.

Results Visualization
The script generates a comparison plot showing actual PM2.5 AQI values, Linear Regression predictions, and SVR predictions based on Ozone AQI values.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset from a CSV file
# Replace 'your_dataset.csv' with the actual filename
# 1)Data Preparation
df = pd.read_csv('global air pollution dataset.csv')

#2)Data cleansing
# Drop rows with NaN values in the target variable 'PM2.5 AQI Value'
df = df.dropna(subset=['PM2.5 AQI Value'])

# Extract the features and target variable
X = df['Ozone AQI Value'].values.reshape(-1, 1)  # Feature
y = df['PM2.5 AQI Value'].values  # Target variable

# Manually split the data into training and testing sets
split_index = int(0.8 * len(X))  # Use 80% of the data for training

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

#    - - - -    #

# Linear Regression without using fit method
#Finsing values for to substitue in formulae : :
X_mean = np.mean(X_train)
y_mean = np.mean(y_train)

numerator = np.sum((X_train - X_mean) * (y_train - y_mean))
denominator = np.sum((X_train - X_mean)**2)

slope_lr = numerator / denominator
intercept_lr = y_mean - slope_lr * X_mean

# Make predictions on the test set using Linear Regression
y_pred_lr = slope_lr * X_test.flatten() + intercept_lr  # Flatten X_test to match the shape of y_test

# Support Vector Regression
svr_model = SVR(kernel='rbf')  # You can choose different kernels and hyperparameters
svr_model.fit(X_train, y_train)  # Use scaled features for SVR

# Make predictions on the test set using SVR
y_pred_svr = svr_model.predict(X_test)

# Evaluate the models using Mean Absolute Error
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)

# Evaluate the models using R-squared coefficient
r2_lr = r2_score(y_test, y_pred_lr)
r2_svr = r2_score(y_test, y_pred_svr)

# Print the results
print("Linear Regression Metrics:")
print("Mean Absolute Error:", mae_lr)
print("R-squared:", r2_lr)

print("\nSupport Vector Regression Metrics:")
print("Mean Absolute Error:", mae_svr)
print("R-squared:", r2_svr)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='black', label='Actual')
plt.plot(X_test, y_pred_lr, color='blue', linewidth=3, label='Linear Regression Line')
plt.plot(X_test, y_pred_svr, color='red', linewidth=3, label='SVR Prediction')
plt.xlabel('Ozone AQI Value')
plt.ylabel('PM2.5 AQI Value')
plt.title('Comparison of Predictions')
plt.legend()
plt.show()
