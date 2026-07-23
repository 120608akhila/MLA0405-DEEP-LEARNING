import numpy as np

np.random.seed(10)

true_mean = 5
true_std = 2

data = np.random.normal(true_mean, true_std, 1000)

estimated_mean = np.mean(data)
estimated_variance = np.var(data)

print("Actual Mean :", true_mean)
print("Estimated Mean :", round(estimated_mean, 3))

print("Actual Variance :", true_std ** 2)
print("Estimated Variance :", round(estimated_variance, 3))
