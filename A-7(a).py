import numpy as np

# Input values
inputs = np.array([1, 2])

# Weights
weights = np.array([[0.5, 0.2],
                    [0.3, 0.8]])

# Bias
bias = np.array([0.1, 0.2])

# Forward propagation
weighted_sum = np.dot(inputs, weights) + bias

# Sigmoid activation
output = 1 / (1 + np.exp(-weighted_sum))

print("Weighted Sum:", weighted_sum)
print("Network Output:", output)
