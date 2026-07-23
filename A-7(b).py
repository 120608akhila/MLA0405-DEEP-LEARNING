import numpy as np

# Input and target
X = np.array([1, 2])
target = 1

# Initial weights
weights = np.array([0.5, 0.3])
bias = 0.1

# Learning rate
lr = 0.1

# Forward propagation
z = np.dot(X, weights) + bias
output = 1 / (1 + np.exp(-z))

# Error
error = target - output

# Sigmoid derivative
delta = error * output * (1 - output)

# Update weights
weights = weights + lr * delta * X
bias = bias + lr * delta

print("Updated Weights:", weights)
print("Updated Bias:", round(bias, 4))
