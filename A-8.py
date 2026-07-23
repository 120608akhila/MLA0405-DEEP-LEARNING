import matplotlib.pyplot as plt

learning_rates = [0.01, 0.1, 0.5]

for lr in learning_rates:
    x = 5
    values = []

    for i in range(20):
        gradient = 2 * x
        x = x - lr * gradient
        values.append(x)

    plt.plot(values, label=f"LR={lr}")

plt.title("Gradient Descent")
plt.xlabel("Iterations")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
