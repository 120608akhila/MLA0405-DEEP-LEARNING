import numpy as np
import matplotlib.pyplot as plt

# Dataset
X = np.array([1,2,3,4,5], dtype=float)
Y = np.array([2,4,6,8,10], dtype=float)

m = 0
b = 0

learning_rate = 0.01
epochs = 1000

n = len(X)
losses = []

for i in range(epochs):

    Y_pred = m*X + b

    loss = np.mean((Y - Y_pred)**2)

    losses.append(loss)

    dm = (-2/n) * np.sum(X*(Y-Y_pred))
    db = (-2/n) * np.sum(Y-Y_pred)

    m = m - learning_rate*dm
    b = b - learning_rate*db

print("Slope =", m)
print("Intercept =", b)

plt.plot(losses)

plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Learning Curve")

plt.show()
