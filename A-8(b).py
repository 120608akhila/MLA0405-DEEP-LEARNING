from sklearn.linear_model import SGDRegressor
from sklearn.datasets import make_regression

# Generate regression dataset
X, y = make_regression(
    n_samples=200,
    n_features=1,
    noise=5,
    random_state=42
)

# Create and train model
model = SGDRegressor(max_iter=1000, random_state=42)
model.fit(X, y)

# Display coefficient
print("Coefficient :", round(model.coef_[0], 4))
