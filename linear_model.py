# linear_model.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class LinearModel(ABC):
    """
    Abstract base class for linear models.
    """

    def __init__(self, learning_rate: float = 0.01, l1_reg: float = 0.0, l2_reg: float = 0.0, verbose: bool = False):
        """
        Initializes the model's hyperparameters.

        Args:
            learning_rate (float): The learning rate for gradient descent.
            l1_reg (float): L1 regularization coefficient.
            l2_reg (float): L2 regularization coefficient.
            verbose (bool): Flag to enable verbose output during training.
        """
        self.learning_rate = learning_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.verbose = verbose
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000):
        """
        Trains the model on the provided data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            epochs (int): Number of training epochs.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for the given data.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted target values.
        """
        pass

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the loss function.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            float: Loss value.
        """
        raise NotImplementedError("This method must be overridden in a subclass.")

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Computes the performance metrics.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            dict: Dictionary containing the performance metrics.
        """
        raise NotImplementedError("This method must be overridden in a subclass.")
