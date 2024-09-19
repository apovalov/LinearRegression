import numpy as np
from typing import Optional
from linear_model import LinearModel

class LinearRegression(LinearModel):
    """
    Linear Regression class.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, epoches: int = 1000):
        """
        Trains the model using gradient descent.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            epochs (int): Number of training epochs.
        """

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_samples)
        self.bias = 0.0

        for epoch in range(epoches):
            y_pred = X.dot(self.weights) + self.bias
            error = y_pred - y

            # Calculate gradients
            dw  = (1 / n_samples) * X.dot(error) + self.l1_reg * np.sign(self.weights) + self.l2_reg * self.weights
            db = (1 / n_samples) * np.sum(error)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if self.verbose and epoch % 100 == 0:
                loss = self._compute_loss(y, y_pred)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target values.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted target values.
        """
        return X.dot(self.weights) + self.bias


    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the Mean Squared Error (MSE).

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            float: MSE value.
        """
        return np.mean((y_true - y_pred) ** 2)


    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Computes regression performance metrics.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            dict: Dictionary containing performance metrics.
        """
        mse = self._compute_loss(y_true, y_pred)
        return {"MSE": mse}