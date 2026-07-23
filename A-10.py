import numpy as np
import matplotlib.pyplot as plt

# Different dimensions
dimensions = [2, 5, 10, 20, 50, 100]

average_distance = []

# Calculate average distance
for d in dimensions:
    points = np.random.rand(100, d)

    distances = np.linalg.norm(points[0] - points[1:], axis=1)

    average_distance.append(np.mean(distances))

# Display values
for d, dist in zip(dimensions, average_distance):
    print("Dimension:", d, " Average Distance:", round(dist, 4))

# Plot graph
plt.plot(dimensions, average_distance, marker='o')
plt.title("Curse of Dimensionality")
plt.xlabel("Dimensions")
plt.ylabel("Average Distance")
plt.grid(True)
plt.show()
