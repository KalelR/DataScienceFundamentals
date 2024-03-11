import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from data_generation import generate_linear_noisy_data

    
def estimate_slope(x, y):
    y_mean = np.mean(y)
    x_mean = np.mean(x)
    numerator = np.dot(x - x_mean, y - y_mean)
    denominator = np.sum( (x - x_mean)**2 )
    return numerator / denominator

def linear_regression_coeffs(x, y):
    y_mean = np.mean(y)
    x_mean = np.mean(x)
    slope = estimate_slope(x, y)
    intercept = y_mean - slope*x_mean 
    return intercept, slope

def linear_regression_estimate(x, intercept, slope):
    return intercept + slope * x

def linear_regression_coeffs_scikit(X, y):
    """
    Perform linear regression analysis on the given dataset.
    
    Parameters:
    X (array-like): Independent variable.
    y (array-like): Dependent variable.
    
    Returns:
    slope (float): Slope of the fitted line.
    intercept (float): Intercept of the fitted line.
    """
    
    # Reshape X into a 2D array (required by scikit-learn)
    X = np.array(X).reshape(-1, 1)
    
    # Perform linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Get the slope and intercept of the fitted line
    slope = model.coef_[0]
    intercept = model.intercept_
    
    return intercept, slope


num_points = 100
x_amp = 10
noise_strength = 2
slope = 3
intercept = 2
x, y_true, y = generate_linear_noisy_data(num_points, x_amp, noise_strength, slope, intercept)

#perform L.R.
intercept_est, slope_est = linear_regression_coeffs(x, y)
y_estimate = linear_regression_estimate(x, intercept_est, slope_est)

intercept_sci, slope_sci = linear_regression_coeffs_scikit(x, y) 
y_estimate_sci = linear_regression_estimate(x, intercept_sci, slope_sci)

# Plot the data
plt.scatter(x, y, label='Data points')
plt.scatter(x, y_estimate, label=f"Estimated data y = {intercept_est:.2f} + {slope_est:.2f}x", c="blue")
plt.scatter(x, y_estimate_sci, label=f"Estimated data using SciKit y = {intercept_sci:.2f} + {slope_sci:.2f}x", c="red")
plt.scatter(x, y_true, label=f"True noiseless data, y = {intercept:.2f} + {slope:.2f}x", c="black")
plt.xlabel('X')
plt.ylabel('y')
plt.title('Randomly Generated Dataset for Linear Regression')
plt.legend()
plt.grid(True)
plt.show()


