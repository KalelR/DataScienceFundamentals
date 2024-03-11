# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from prettytable import PrettyTable
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
%matplotlib inline


df = pd.read_csv("Advertising.csv")

# Function to fit a linear model on the predictor passed as a parameter, compute the parameters
# and plot the fit of the R^2
def fit_and_plot_linear(x):

	# Split the data into train and test sets with train size of 0.8
	# Set the random state as 0 to get reproducible results
	x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=0)

	# Initialize a LinearRegression object
	lreg = LinearRegression()

	# Fit the model on the train data
	lreg.fit(x_train, y_train)

	# Predict the response variable of the train set using the trained model
	y_train_pred = lreg.predict(x_train)

	# Predict the response variable of the test set using the trained model
	y_test_pred= lreg.predict(x_test)

	# Compute the R-square for the train predictions
	r2_train = r2_score(y_train, y_train_pred)

	# Compute the R-square for the test predictions
	r2_test = r2_score(y_test, y_test_pred)

	# Code to plot the prediction for the train and test data
	plt.scatter(x_train, y_train, color='#B2D7D0', label = "Train data")
	plt.scatter(x_test, y_test, color='#EFAEA4', label = "Test data")
	plt.plot(x_train, y_train_pred, label="Train Prediction", color='darkblue', linewidth=2)
	plt.plot(x_test, y_test_pred, label="Test Prediction", color='k', alpha=0.8, linewidth=2, linestyle='--')
	name = x.columns.to_list()[0]
	plt.title(f"Plot to indicate linear model predictions")
	plt.xlabel(f"{name}", fontsize=14)
	plt.ylabel("Sales", fontsize=14)
	plt.legend()
	plt.show()

	# Return the r-square of the train and test data
	return r2_train, r2_test



# Function to fit a multilinear model on all the predictors in the dataset passed as a parameter, compute the parameters
# and plot the fit of the R^2
def fit_and_plot_multi():

	# Get the predictor variables
	x = df[['TV','Radio','Newspaper']]

	# Split the data into train and test sets with train size of 0.8
	# Set the random state as 0 to get reproducible results
	x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=0)

	# Initialize a LinearRegression object to perform Multi-linear regression
	lreg = LinearRegression()

	# Fit the model on the train data
	lreg.fit(x_train, y_train)

	# Predict the response variable of the train set using the trained model
	y_train_pred = lreg.predict(x_train)

	# Predict the response variable of the test set using the trained model
	y_test_pred= lreg.predict(x_test)

	# Compute the R-square for the train predictions
	r2_train = r2_score(y_train, y_train_pred)

	# Compute the R-square for the test predictions
	r2_test = r2_score(y_test, y_test_pred)

	# Return the r-square of the train and test data
	return r2_train, r2_test




df_results = pd.DataFrame(columns=['Predictor', 'R2 Train', 'R2 Test'])

predictor_list = df.columns.tolist()[0:-1]
for predictor in predictor_list:
    r2_train, r2_test = fit_and_plot_linear(df[[predictor]])
    df_results = df_results.append({'Predictor':predictor, 'R2 Train': r2_train, 'R2 Test': r2_test}, ignore_index=True)

print(df_results)


# Initialize a list to store the MSE values
mse_list = []

# Create a list of lists of all unique predictor combinations
# For example, if you have 2 predictors,  A and B, you would 
# end up with [['A'],['B'],['A','B']]
cols = [['Radio'], ['Newspaper'], ['TV'], ['Radio', 'Newspaper'], ['Radio', 'TV'], ['Newspaper', 'TV'], ['Radio', 'Newspaper', 'TV']]

# Loop over all the predictor combinations 
for i in cols:
    # Set each of the predictors from the previous list as x
    X = df[i]
    
    # Set the "Sales" column as the reponse variable
    y = df['Sales']
   
    # Split the data into train-test sets with 80% training data and 20% testing data. 
    # Set random_state as 0
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

    # Initialize a Linear Regression model
    lreg = LinearRegression()

    # Fit the linear model on the train data
    fit = lreg.fit(x_train, y_train)

    # Predict the response variable for the test set using the trained model
    y_pred = lreg.predict(x_test)
    
    # Compute the MSE for the test data
    MSE = mean_squared_error(y_test, y_pred)
    
    # Append the computed MSE to the initialized list
    mse_list.append(MSE)


# Helper code to display the MSE for each predictor combination
t = PrettyTable(['Predictors', 'MSE'])

for i in range(len(mse_list)):
    t.add_row([cols[i],round(mse_list[i],3)])

print(t)