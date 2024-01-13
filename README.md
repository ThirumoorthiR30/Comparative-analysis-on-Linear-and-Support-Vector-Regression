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
