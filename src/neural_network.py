import logging

import numpy as np
from scipy.special import expit  # Better numerical stability

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
    """
    A simple multi-layer perceptron (MLP) neural network classifier implemented
    from scratch.

    This class is designed to mimic a simplified/modified API of scikit-learn's
    MLPClassifier. It supports multiple activation functions, mini-batch
    stochastic gradient descent (SGD), L2 regularization, and early stopping.

    Parameters
    ----------
    layer_sizes : tuple of int, default=(784, 100, 100, 10)
        The number of neurons in each layer, including input and output layers.
        Example: (784, 100, 10) defines a network with 784 input features, one
        hidden layer of 100 neurons, and 10 output classes.

    activation_function : str, default='relu'
        The activation function to use for all hidden layers. Options: -
        'identity': No transformation (linear) - 'logistic': Sigmoid function -
        'tanh': Hyperbolic tangent - 'relu': Rectified Linear Unit

    alpha : float, default=1e-4
        L2 regularization (weight decay) strength. Higher values increase
        regularization.

    batch_size : int or 'auto', default='auto'
        Mini-batch size for stochastic gradient descent. If 'auto', uses
        min(200, n_samples).

    learning_rate : float, default=0.001
        Step size for weight updates during gradient descent.

    max_iter : int, default=2000
        Maximum number of training epochs (full passes through the dataset).

    tolerance : float, default=1e-4
        Minimum loss improvement required to continue training. Training stops
        if improvement is below this threshold for `n_iter_no_change`
        consecutive epochs.

    n_iter_no_change : int, default=10
        Number of epochs with insufficient improvement before triggering early
        stopping.

    Attributes
    ----------
    weights : list of ndarray
        Learned weight matrices for each layer. weights[i] has shape (n_in,
        n_out).

    biases : list of ndarray
        Learned bias vectors for each layer. biases[i] has shape (n_out,).

    layer_sizes : tuple of int
        The architecture specification passed during initialization.

    n_iter : int
        Actual number of epochs completed during training.

    loss_curve : ndarray
        Training loss recorded after each epoch.

    validation_scores : ndarray
        Validation accuracy scores recorded after each epoch, if tracking was
        enabled by passing `validation_data=(x_val, y_val)` to `fit()`.

    activation_function, d_activation_function : callable
        The activation function and its derivative, selected from the global
        mappings.

    activation_func_name : str
        Name of activation function passed during initialization.

    Methods
    -------
    fit(x_train, y_train, validation_data=(x_test, y_test))
        Train the network using mini-batch SGD with optional early stopping. Set
        `validation_data`, otherwise accuracy scores will not be tracked in
        `validation_scores`.

    predict(x)
        Predict class label for input x.

    predict_proba(x)
        Return probability estimates for each class.

    score(x, y)
        Compute mean classification accuracy on given data and labels.

    Notes
    -----
    - Uses Xavier/Glorot initialization for logistic/tanh activations and He
      initialization for ReLU to promote stable gradient flow during training.
    - Output layer uses softmax-like behavior via cross-entropy loss; for
      multi-class classification, ensure the final layer size matches the number
      of classes.
    - Labels `y` must be provided as one-hot encoded vectors (shape (n_samples,
      n_classes)).
    - Early stopping monitors training loss; for validation-based stopping, pass
      `validation_data=(x_val, y_val)` to `fit()`.
    - No momentum or adaptive learning rates are implemented; this is vanilla
      SGD.

    References
    ----------
    .. [1] scikit-learn MLPClassifier:
        https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

    Examples
    --------
    >>> net = NeuralNetwork()
    >>> # Load datasets
    >>> train_images = load_mnist_images("dataset/train-images-idx3-ubyte")
    >>> train_labels = load_mnist_labels("dataset/train-labels-idx1-ubyte")
    >>> test_images = load_mnist_images("dataset/t10k-images-idx3-ubyte")
    >>> test_labels = load_mnist_labels("dataset/t10k-labels-idx1-ubyte")
    >>> # Format data
    >>> x_train, y_train = format_mnist_data(train_images, train_labels)
    >>> x_val, y_val = format_mnist_data(test_images, test_labels)
    >>> net.fit(x_train, y_train, validation_data=(x_val, y_val))
    >>> # Evaluate
    >>> final_accuracy = net.score(X_test, y_test)
    >>> predictions = net.predict(X_test)
    >>> probabilities = net.predict_proba(X_test)
    >>> # Plot training progress
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(net.loss_curve, label="Loss")
    >>> plt.plot(net.validation_scores, label="Accuracy of Test Dataset")
    >>> plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Curve')
    >>> plt.legend()
    >>> plt.show()
    """

    def __init__(
        self,
        layer_sizes: tuple[int] = (784, 100, 100, 10),
        activation_function: str = "relu",
        *,
        alpha: float = 1e-4,
        batch_size: int = "auto",
        learning_rate: float = 0.001,
        max_iter: int = 2000,
        tolerance: float = 1e-4,
        n_iter_no_change: int = 25,
    ):

        # TODO: Add parameter value validation. Low priority.

        # MLP functions, weights, biases
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
        self.activation_func_name = activation_function

        # Validation score tracking
        self.validation_scores = np.array([])

    def _get_initial_weights(self, layer_sizes):
        """Initialize weights using Xavier/Glorot initialization (He for ReLU)."""
        weights = []
        for i in range(len(layer_sizes) - 1):
            if self.activation_func_name == "relu":
                # He initialization for ReLU
                scale = np.sqrt(2.0 / layer_sizes[i])
            else:
                # Xavier initialization for logistic/tanh
                scale = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i + 1]))
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            weights.append(w)
        return weights

    def _get_initial_biases(self, layer_sizes):
        """Initialize biases to zeros."""
        return [np.zeros(size) for size in layer_sizes[1:]]

    def _forward_pass(self, x):
        """Forward pass returning activations and pre-activations for backprop."""

        activations = [x]
        pre_activations = []

        a = x
        for w, b in zip(weights, biases):
            z = a @ w + b
            pre_activations.append(z)
            a = self.activation_function(z)
            activations.append(a)

        return activations, pre_activations

    def _compute_loss(self, y_true, y_pred, weights=None):
        """Compute cross-entropy loss with optional L2 regularization."""
        n_samples = y_true.shape[0]

        # Clip to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        cross_entropy = -np.sum(y_pred * np.log(y_pred_clipped)) / n_samples

        # L2 regularization term
        if self.alpha > 0 and weights is not None:
            l2_term = 0.5 * self.alpha * sum(np.sum(w**2) for w in weights)
            cross_entropy += l2_term

        return cross_entropy

    def _backward_pass(self, x, y_true, activations, pre_activations):
        """Backward pass computing gradients for all weights and biases."""
        n_samples = x.shape[0]

        # Output layer delta (cross-entropy + activation derivative)
        if self._activation_name == "logistic":
            # Simplified gradient for logistic + cross-entropy
            delta = activations[-1] - y_true
        else:
            d_loss_da = activations[-1] - y_true
            delta = d_loss_da * self.d_activation_function(pre_activations[-1])

        gradients_w = []
        gradients_b = []

        # Backpropagate through layers (reverse order)
        for l in reversed(range(len(self.weights))):
            grad_w = activations[l].T @ delta / n_samples
            grad_b = np.mean(delta, axis=0)

            # Add L2 regularization gradient
            if self.alpha > 0:
                grad_w += self.alpha * self.weights[l]

            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)

            if l > 0:
                delta = (delta @ self.weights[l].T) * self.d_activation_function(
                    pre_activations[l - 1]
                )

        return gradients_w, gradients_b

    def _update_weights(self, gradients_w, gradients_b):
        """Apply gradient descent update to weights and biases."""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

    def _get_batch_indices(self, n_samples, batch_size, rng):
        """Yield shuffled mini-batch indices."""
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            yield indices[start:end]

    def fit(self, x, y, validation_data=None):
        """Train the neural network using mini-batch SGD with early stopping."""
        n_samples = x.shape[0]
        batch_size = (
            min(200, n_samples) if self.batch_size == "auto" else self.batch_size
        )

        x_train, y_train = x, y

        # Handle validation
        if validation_data is not None:
            x_val, y_val = validation_data
        else:
            x_val, y_val = None, None

        # Reset training state
        self.loss_curve = np.array([])
        self.validation_scores = np.array([])

        best_loss = np.inf
        no_improvement = 0
        rng = np.random.RandomState(42)

        for iteration in range(self.max_iter):
            self.n_iter = iteration + 1

            # Shuffle and mini-batch SGD
            perm = np.arange(len(x_train))
            rng.shuffle(perm)
            x_shuf, y_shuf = x_train[perm], y_train[perm]

            for batch_idx in self._get_batch_indices(len(x_train), batch_size, rng):
                x_batch, y_batch = x_shuf[batch_idx], y_shuf[batch_idx]

                activations, pre_activations = self._forward_pass(x_batch)
                grad_w, grad_b = self._backward_pass(
                    x_batch, y_batch, activations, pre_activations
                )
                self._update_weights(grad_w, grad_b)

            # Track training loss
            activations, _ = self._forward_pass(x_train)
            train_loss = self._compute_loss(y_train, activations[-1])
            self.loss_curve = np.append(self.loss_curve, train_loss)

            # Track validation score if requested
            if x_val is not None:
                val_score = self.score(x_val, y_val)
                self.validation_scores = np.append(self.validation_scores, val_score)

            # Early stopping check
            if abs(best_loss - train_loss) < self.tolerance:
                no_improvement += 1
                if no_improvement >= self.n_iter_no_change:
                    logger.info(f"Early stopping at iteration {iteration + 1}")
                    break
            else:
                no_improvement = 0
                best_loss = min(best_loss, train_loss)

    def predict_proba(self, x):
        """Return probability estimates for each class."""
        activations, _ = self._forward_pass(x)
        return activations[-1]

    def predict(self, x):
        """Predict class label for x."""
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=1)

    def score(self, x, y):
        """Return mean accuracy on test data x and labels y."""

        predictions = [self.predict(xi) for xi in x]

        successes = [pred == true for pred, true in zip(predictions, y)]

        return np.mean(successes)
