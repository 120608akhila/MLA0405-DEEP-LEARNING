import numpy as np

# Input values
x = np.array([2, 3])

# Weights
w = np.array([0.5, 0.8])

# Bias
b = 1

# Weighted sum
z = np.dot(x, w) + b

# Sigmoid activation
sigmoid = 1 / (1 + np.exp(-z))

# ReLU activation
relu = max(0, z)

# Display results
print("Weighted Sum :", round(z, 2))
print("Sigmoid Output :", round(sigmoid, 4))
print("ReLU Output :", round(relu, 2))
