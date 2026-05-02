import logging

import numpy as np
from scipy.special import expit

logger = logging.getLogger(__name__)


ACTIVATION_FUNCTIONS = {
    "identity": lambda x: x,
    "logistic": lambda x: expit(x),
    "tanh": lambda x: np.tanh(x),
    "relu": lambda x: np.maximum(0.0, x),
}

D_ACTIVATION_FUNCTIONS = {
    "identity": lambda x: np.ones_like(x),
    "logistic": lambda x: expit(x) * (1 - expit(x)),
    "tanh": lambda x: 1 - np.tanh(x) ** 2,
    "relu": lambda x: np.where(x > 0, 1, 0),
}


class NeuralNetwork:

    # Designed to somewhat mimic API of sklearn MLP, with many simplifications:
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

    def __init__(
        self,
        layer_sizes: tuple[int] = (
            784,
            100,
            100,
            10,
        ),
        activation_function: str = "logistic",
        *,
        track_validation_score: bool = False,
        alpha: float = 1e-4,
        batch_size: int = "auto",
        learning_rate: float = 0.001,
        max_iter: int = 2000,
        tolerance: float = 1e-4,
        n_iter_no_change: int = 10,
    ):

        # TODO: Add parameter value validation. Low priority.

        # MLP structure
        self.activation_function = ACTIVATION_FUNCTIONS[activation_function]
        self.d_activation_function = D_ACTIVATION_FUNCTIONS[activation_function]

        self.weights = self._get_initial_weights(layer_sizes)
        self.biases = self._get_initial_biases(layer_sizes)

        # L2 regularization
        self.alpha = alpha

        # Training parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Training stop conditions
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.n_iter_no_change = n_iter_no_change

        # Information about MLP object
        self.layer_sizes = layer_sizes
        self.n_iter = 0
        self.loss_curve = np.array([])

        self.validation_scores = None

        if track_validation_score:
            self.validation_scores = np.array([])

    def fit(self, x, y):
        # SGD, no Nesterov momentum
        # batch_size=min(200, n_samples)
        # if track_validation_score: split x and y into training and validation and score validation every epoch and store
        return self # Does this make sense?

    def fit_run_iteration(self, x, y):
        pass

    def predit(self, x, y):
        pass

    def predit_proba(self, x, y):
        pass

    def score(self, x, y):
        pass

    def _get_initial_weights(self, layer_sizes):
        pass

    def _get_initial_biases(self, layer_sizes):
        pass


