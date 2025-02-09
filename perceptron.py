# Re-import necessary libraries since execution state was reset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

class Perceptron:
    # Initialize hyperparameters (learning rate and number of iterations)
    def __init__(self, eta=0.1, T=10):
        self.eta = eta
        self.T = T
        self.history = []  # Store decision boundary parameters for visualization

    def fit(self, X, y):
        # Extract dimensions properly
        N, d = X.shape  # N = num_samples, d = num_features

        # Randomly initialize weights using numpy
        self.beta = np.random.uniform(-1.0, 1.0, size=d)  # Correct size
        self.beta0 = np.random.uniform(-1.0, 1.0)  # Bias initialization

        # Track the number of misclassifications per iteration
        self.errors_ = []

        # Iterate over the dataset for T epochs
        for t in range(self.T):
            errors = 0
            for xi, yi in zip(X, y):
                if yi * (np.dot(xi, self.beta) + self.beta0) <= 0:
                    # Store pre-update decision boundary
                    self.history.append((self.beta.copy(), self.beta0))

                    # Update weights
                    self.beta += self.eta * yi * xi
                    self.beta0 += self.eta * yi

                    # Store post-update decision boundary
                    self.history.append((self.beta.copy(), self.beta0))

                    errors += 1  # Count misclassification
            self.errors_.append(errors)
        
        return self

# Prepare data
X = df.iloc[:, [0,1,2]].values  # Use first three features
y = df.iloc[:, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)  # Convert labels to -1 and 1

# Train Perceptron model
model = Perceptron(T=10)
model.fit(X, y)

# Plot error rate over iterations
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(model.errors_) + 1), model.errors_, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Number of Errors")
plt.title("Perceptron Convergence")
plt.grid(True)
plt.show()

# Plot 3D decision boundaries before and after each update
x_vals = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 10)
y_vals = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 10)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

for i, (beta, beta0) in enumerate(model.history):
    Z_grid = (-beta[0] * X_grid - beta[1] * Y_grid - beta0) / beta[2]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of data points
    ax.scatter(X[y == -1, 0], X[y == -1, 1], X[y == -1, 2], color='blue', label='Class -1')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], color='red', label='Class 1')

    # Decision boundary
    ax.plot_surface(X_grid, Y_grid, Z_grid, color='gray', alpha=0.5)

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    ax.set_title(f"Decision Boundary at Step {i+1}")
    ax.legend()

    plt.savefig(f"decision_boundary_step_{i+1}.png",bbox_inches='tight', pad_inches=0)
    plt.close()
