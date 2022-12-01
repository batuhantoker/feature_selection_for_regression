import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Define the input and output data
X = np.random.randn(1000, 10)
y = X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3] + X[:, 4] + X[:, 5] + X[:, 6] + X[:, 7] + X[:, 8] + X[:, 9]

# Select the best features using trial and error
best_mse = float("inf")
best_features = []
for i in range(1, 10):
    for features in combinations(range(10), i):
        cur_X = X[:, features]
        cur_mse = 0
        for train_idx, test_idx in KFold(10).split(cur_X):
            X_train, X_test = cur_X[train_idx], cur_X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train the MLP regression model
            model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000)
            model.fit(X_train, y_train)

            # Evaluate the model on the test data
            cur_mse += mean_squared_error(y_test, model.predict(X_test))
        cur_mse /= 10

        # Update the best features and the best MSE
        if cur_mse < best_mse:
            best_mse = cur_mse
            best_features = features

# Print the selected features and the corresponding MSE
print(f"Selected features: {best_features}")
print(f"Test MSE: {best_mse:.4f}")
