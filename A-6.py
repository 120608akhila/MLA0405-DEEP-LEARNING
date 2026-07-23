from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Generate dataset
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# Train Perceptron
model = Perceptron(random_state=42)
model.fit(X, y)

# Prediction
y_pred = model.predict(X)

# Accuracy
accuracy = accuracy_score(y, y_pred)

print("Accuracy:", accuracy)
