# logistic_regression.py
import numpy as np
from typing import Optional
from linear_model import LinearModel


class LogisticRegression(LinearModel):
    """
    Logistic Regression class.
    """

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        Args:
            z (np.ndarray): Input value.

        Returns:
            np.ndarray: Sigmoid function applied to the input.
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000):
        """
        Trains the model using gradient descent.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Vector of class labels.
            epochs (int): Number of training epochs.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(epochs):
            linear_model = X.dot(self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            error = y_pred - y

            # Calculate gradients
            dw = (1 / n_samples) * X.T.dot(error) + self.l1_reg * np.sign(self.weights) + self.l2_reg * self.weights
            db = (1 / n_samples) * np.sum(error)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if self.verbose and epoch % 100 == 0:
                loss = self._compute_loss(y, y_pred)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Vector of predicted class labels.
        """
        linear_model = X.dot(self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        return (y_pred >= 0.5).astype(int)

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the logistic loss function.

        Args:
            y_true (np.ndarray): Vector of true class labels.
            y_pred (np.ndarray): Vector of predicted probabilities.

        Returns:
            float: Loss value.
        """
        epsilon = 1e-15  # To avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Computes classification performance metrics.

        Args:
            y_true (np.ndarray): Vector of true class labels.
            y_pred (np.ndarray): Vector of predicted class labels.

        Returns:
            dict: Dictionary containing performance metrics.
        """
        accuracy = np.mean(y_true == y_pred)
        return {"Accuracy": accuracy}
