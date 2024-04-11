import numpy as np
import matplotlib.pyplot as plt
from data_generation import generate_linear_noisy_data
from sklearn.neighbors import KNeighborsRegressor

# Generate data
num_points = 100
x_amp = 10
noise_strength = 2
slope = 3
intercept = 2
x, y_true, y = generate_linear_noisy_data(num_points, x_amp, noise_strength, slope, intercept)
X = x.reshape(-1, 1) #convert into 2d array with one column

X_new = np.linspace(np.min(x), np.max(x), 100).reshape(-1, 1) #create new dataset for prediction



# # Perform kNN estimation
# k = 5  # Number of neighbors
# knn_model = KNeighborsRegressor(n_neighbors=k)
# knn_model.fit(X, y)
# 
# # Randomly select a point for estimation
# point_to_estimate = np.array([[5]])  # Replace with your desired point

# Perform kNN estimation for the selected point
# estimated_value = knn_model.predict(point_to_estimate)

plt.scatter(X, y, label='Original data')
plt.scatter(X, y_true, label='Original noiseless data', color='black')

k_list = [1, 5, 10]
colors = ['red', 'orange', 'brown']
for idx, k in enumerate(ks):
    knn_model = KNeighborsRegressor(n_neighbors=k)
    knn_model.fit(X, y)
    y_pred = knn_model.predict(X_new)
    plt.scatter(X_new, y_pred, color=colors[idx], label=f"Estimated values with k = {k}")

plt.xlabel('X')
plt.ylabel('y')
plt.title('kNN Estimation')
plt.legend()
plt.grid(True)
plt.show()
