import numpy as np
#generate data
def generate_linear_noisy_data(num_points, x_amp, noise_strength, slope, intercept):
    # Set random seed for reproducibility
    np.random.seed(1)
    
    # Generate random x values between 0 and 10
    x = x_amp * np.random.rand(num_points)
    
    # Generate random noise for y values
    noise = noise_strength * np.random.randn(num_points)
    
    # Generate y values using a linear relationship with noise
    y_noise = slope * x + intercept + noise
    y_true = slope * x + intercept
    return x, y_true, y_noise