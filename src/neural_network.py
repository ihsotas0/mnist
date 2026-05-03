import logging
import numpy as np
from scipy.special import expit, softmax

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
    "relu": lambda x: np.where(x > 0, 1, 0).astype(float),
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
        regularization. Set to 0 to disable regularization

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
    fit(x_train, y_train, validation_data=(x_val, y_val))
        Train the network using mini-batch SGD with optional early stopping. Set
        `validation_data`, otherwise accuracy scores will not be tracked in
        `validation_scores`.

    predict(x)
        Predict class label for input x.

    predict_proba(x)
        Return probability estimates for each class.

    score(x_test, y_test)
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
        batch_size: int | str = "auto",
        learning_rate: float = 0.001,
        max_iter: int = 200,
        tolerance: float = 1e-4,
        n_iter_no_change: int = 25,
        random_state: int | None = None,
    ):

        # FUTURE JONAH. START CODE REVIEW FROM HERE DOWN

        logger.info(f"Initializing NeuralNetwork with layer_sizes={layer_sizes}")
        logger.debug(
            f"Parameters: activation={activation_function}, alpha={alpha}, "
            f"batch_size={batch_size}, lr={learning_rate}, max_iter={max_iter}"
        )

        # Parameter validation
        if activation_function not in ACTIVATION_FUNCTIONS:
            logger.error(
                f"Invalid activation_function: '{activation_function}'. "
                f"Valid options: {list(ACTIVATION_FUNCTIONS.keys())}"
            )
            raise ValueError(f"Unknown activation function: {activation_function}")

        if len(layer_sizes) < 2:
            logger.error(
                f"layer_sizes must have at least 2 elements (input and output), got {len(layer_sizes)}"
            )
            raise ValueError("layer_sizes must define at least input and output layers")

        if any(size <= 0 for size in layer_sizes):
            logger.error(f"All layer sizes must be positive, got {layer_sizes}")
            raise ValueError("Layer sizes must be positive integers")

        # MLP functions
        self.activation_func_name = activation_function
        self.activation_function = ACTIVATION_FUNCTIONS[activation_function]
        self.d_activation_function = D_ACTIVATION_FUNCTIONS[activation_function]
        logger.debug(f"Selected activation: {activation_function}")

        # Initialize weights with appropriate scheme
        self.weights = self._get_initial_weights(layer_sizes)
        self.biases = self._get_initial_biases(layer_sizes)
        logger.debug(
            f"Initialized {len(self.weights)} weight matrices with "
            f"{'He' if activation_function == 'relu' else 'Xavier'} initialization"
        )

        # Regularization
        self.alpha = alpha
        if alpha > 0:
            logger.info(f"L2 regularization enabled with alpha={alpha}")

        # Training parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        logger.debug(f"Training config: batch_size={batch_size}, lr={learning_rate}")

        # Stopping conditions
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.n_iter_no_change = n_iter_no_change

        # State tracking
        self.layer_sizes = layer_sizes
        self.n_iter = 0
        self.loss_curve = []  # Use list for efficient appending
        self.validation_scores = []
        self.random_state = random_state

        logger.info(
            f"NeuralNetwork initialized successfully with {sum(w.size for w in self.weights) + sum(b.size for b in self.biases):,} parameters"
        )

    def _get_initial_weights(self, layer_sizes):
        """Initialize weights using Xavier/Glorot initialization (He for ReLU)."""
        weights = []
        for i in range(len(layer_sizes) - 1):
            if self.activation_func_name == "relu":
                scale = np.sqrt(2.0 / layer_sizes[i])
                init_name = "He"
            else:
                scale = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i + 1]))
                init_name = "Xavier"

            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            weights.append(w)
            logger.debug(
                f"Layer {i}->{i+1}: {layer_sizes[i]} x {layer_sizes[i+1]}, "
                f"{init_name} init, scale={scale:.4f}, weight std={np.std(w):.4f}"
            )
        return weights

    def _get_initial_biases(self, layer_sizes):
        """Initialize biases to zeros."""
        biases = [np.zeros(size) for size in layer_sizes[1:]]
        logger.debug(f"Initialized {len(biases)} bias vectors to zeros")
        return biases

    def _forward_pass(self, x):
        """Forward pass returning activations and pre-activations for backprop."""
        logger.debug(f"Forward pass input shape: {x.shape}")

        activations = [x]
        pre_activations = []

        a = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            pre_activations.append(z)

            # Apply activation: use softmax for output layer in multi-class settings
            if i == len(self.weights) - 1 and self.layer_sizes[-1] > 1:
                # Output layer: softmax for numerical stability with cross-entropy
                a = softmax(z, axis=1)
                logger.debug(f"Output layer: applied softmax, shape={a.shape}")
            else:
                a = self.activation_function(z)
                logger.debug(
                    f"Hidden layer {i+1}: activation={self.activation_func_name}, "
                    f"output range=[{a.min():.4f}, {a.max():.4f}]"
                )

            activations.append(a)

        logger.debug(f"Forward pass complete: {len(activations)} activations computed")
        return activations, pre_activations

    def _compute_loss(self, y_true, y_pred, weights=None):
        """Compute cross-entropy loss with optional L2 regularization."""
        n_samples = y_true.shape[0]

        # Numerical stability: clip predictions
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # Cross-entropy: -sum(y * log(y_pred)) / n
        cross_entropy = -np.sum(y_true * np.log(y_pred_clipped)) / n_samples
        logger.debug(f"Cross-entropy loss (unregularized): {cross_entropy:.6f}")

        # L2 regularization term
        l2_term = 0.0
        if self.alpha > 0 and weights is not None:
            l2_term = 0.5 * self.alpha * sum(np.sum(w**2) for w in weights)
            cross_entropy += l2_term
            logger.debug(f"L2 regularization term: {l2_term:.6f}")

        total_loss = cross_entropy
        logger.debug(f"Total loss: {total_loss:.6f}")
        return total_loss

    def _backward_pass(self, x, y_true, activations, pre_activations):
        """Backward pass computing gradients for all weights and biases."""
        n_samples = x.shape[0]
        logger.debug(f"Backward pass: n_samples={n_samples}")

        # Output layer delta: simplified gradient for softmax + cross-entropy
        # This works because d(CE)/d(z) = softmax(z) - y_true when using softmax output
        output_activation = activations[-1]
        delta = output_activation - y_true
        logger.debug(
            f"Output delta: shape={delta.shape}, mean={np.mean(delta):.6f}, "
            f"std={np.std(delta):.6f}"
        )

        gradients_w = []
        gradients_b = []

        # Backpropagate through layers (reverse order)
        for l in reversed(range(len(self.weights))):
            grad_w = activations[l].T @ delta / n_samples
            grad_b = np.mean(delta, axis=0)

            # Add L2 regularization gradient
            if self.alpha > 0:
                grad_w += self.alpha * self.weights[l]
                logger.debug(f"Layer {l}: added L2 gradient (alpha={self.alpha})")

            # Log gradient statistics for debugging
            logger.debug(
                f"Layer {l} gradients: W shape={grad_w.shape}, "
                f"||W||={np.linalg.norm(grad_w):.4f}, "
                f"||b||={np.linalg.norm(grad_b):.4f}"
            )

            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)

            # Propagate error to previous layer (if not input layer)
            if l > 0:
                delta = (delta @ self.weights[l].T) * self.d_activation_function(
                    pre_activations[l - 1]
                )
                logger.debug(f"Propagated delta to layer {l-1}: shape={delta.shape}")

        logger.debug("Backward pass complete")
        return gradients_w, gradients_b

    def _update_weights(self, gradients_w, gradients_b):
        """Apply gradient descent update to weights and biases."""
        logger.debug(f"Updating weights with learning_rate={self.learning_rate}")

        for i in range(len(self.weights)):
            old_w_norm = np.linalg.norm(self.weights[i])
            old_b_norm = np.linalg.norm(self.biases[i])

            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

            new_w_norm = np.linalg.norm(self.weights[i])
            new_b_norm = np.linalg.norm(self.biases[i])

            logger.debug(
                f"Layer {i}: weight norm {old_w_norm:.4f}→{new_w_norm:.4f}, "
                f"bias norm {old_b_norm:.4f}→{new_b_norm:.4f}"
            )

    def _get_batch_indices(self, n_samples, batch_size, rng):
        """Yield shuffled mini-batch indices."""
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        n_batches = int(np.ceil(n_samples / batch_size))
        logger.debug(
            f"Created {n_batches} mini-batches of size ~{batch_size} from {n_samples} samples"
        )

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            yield indices[start:end]

    def fit(self, x, y, validation_data=None):
        """Train the neural network using mini-batch SGD with early stopping."""
        logger.info(
            f"Starting training: {x.shape[0]} samples, {x.shape[1]} features, "
            f"{y.shape[1]} classes"
        )

        n_samples = x.shape[0]
        batch_size = (
            min(200, n_samples) if self.batch_size == "auto" else self.batch_size
        )

        if batch_size > n_samples:
            logger.warning(
                f"batch_size ({batch_size}) > n_samples ({n_samples}); "
                f"using full-batch gradient descent"
            )
            batch_size = n_samples

        logger.info(
            f"Training configuration: batch_size={batch_size}, "
            f"max_epochs={self.max_iter}, early_stopping_patience={self.n_iter_no_change}"
        )

        x_train, y_train = x, y

        # Handle validation
        x_val, y_val = None, None
        if validation_data is not None:
            x_val, y_val = validation_data
            logger.info(f"Validation data provided: {x_val.shape[0]} samples")

        # Reset training state
        self.loss_curve = []
        self.validation_scores = []
        self.n_iter = 0

        best_loss = np.inf
        no_improvement = 0

        # Initialize RNG with user-provided or default seed
        rng = np.random.RandomState(
            self.random_state if self.random_state is not None else 42
        )
        logger.debug(f"Random state initialized with seed={rng.randint(0, 2**31)}")

        for iteration in range(self.max_iter):
            self.n_iter = iteration + 1

            # Shuffle data each epoch
            perm = np.arange(len(x_train))
            rng.shuffle(perm)
            x_shuf, y_shuf = x_train[perm], y_train[perm]

            epoch_losses = []

            # Mini-batch SGD
            for batch_idx, batch_indices in enumerate(
                self._get_batch_indices(len(x_train), batch_size, rng)
            ):
                x_batch, y_batch = x_shuf[batch_indices], y_shuf[batch_indices]

                activations, pre_activations = self._forward_pass(x_batch)
                grad_w, grad_b = self._backward_pass(
                    x_batch, y_batch, activations, pre_activations
                )
                self._update_weights(grad_w, grad_b)

                # Track batch loss for epoch summary
                batch_loss = self._compute_loss(y_batch, activations[-1], self.weights)
                epoch_losses.append(batch_loss)

                if batch_idx % 10 == 0:
                    logger.info(
                        f"Epoch {self.n_iter}, Batch {batch_idx}: loss={batch_loss:.6f}"
                    )

            # Compute full training loss for monitoring
            train_activations, _ = self._forward_pass(x_train)
            train_loss = self._compute_loss(
                y_train, train_activations[-1], self.weights
            )
            self.loss_curve.append(train_loss)

            # Log epoch summary
            avg_batch_loss = np.mean(epoch_losses) if epoch_losses else train_loss
            logger.info(
                f"Epoch {self.n_iter:4d}/{self.max_iter}: "
                f"train_loss={train_loss:.6f} (batch_avg={avg_batch_loss:.6f}), "
                f"lr={self.learning_rate}"
            )

            # Track validation score if requested
            if x_val is not None:
                # HACK: Only pick random 100 or less from x_val and y_val to speed things up
                idx = rng.choice(
                    np.arange(len(x_val)), min(100, len(x_val)), replace=False
                )
                val_score = self.score(x_val[idx], y_val[idx])
                self.validation_scores.append(val_score)
                logger.info(f"  -> Validation accuracy: {val_score:.4f}")

            # Early stopping check: did loss improve by at least tolerance?
            if best_loss - train_loss > self.tolerance:
                # Significant improvement
                logger.debug(
                    f"Loss improved: {best_loss:.6f} -> {train_loss:.6f} "
                    f"(delta={best_loss - train_loss:.6f} > tolerance={self.tolerance})"
                )
                best_loss = train_loss
                no_improvement = 0
            else:
                # No significant improvement
                no_improvement += 1
                logger.debug(
                    f"No significant improvement (patience: {no_improvement}/{self.n_iter_no_change})"
                )

                if no_improvement >= self.n_iter_no_change:
                    logger.info(
                        f"Early stopping triggered at epoch {self.n_iter}: "
                        f"no improvement for {self.n_iter_no_change} consecutive epochs"
                    )
                    break

        # Training complete summary
        logger.info(
            f"Training completed: {self.n_iter}/{self.max_iter} epochs, "
            f"final loss={self.loss_curve[-1]:.6f}"
        )

        if self.validation_scores:
            best_val_idx = np.argmax(self.validation_scores)
            logger.info(
                f"Best validation accuracy: {max(self.validation_scores):.4f} at epoch {best_val_idx + 1}"
            )

        # Convert lists to arrays for API compatibility
        self.loss_curve = np.array(self.loss_curve)
        self.validation_scores = np.array(self.validation_scores)

        return self

    def predict_proba(self, x):
        """Return probability estimates for each class."""
        logger.debug(f"Predicting probabilities for {x.shape[0]} samples")
        activations, _ = self._forward_pass(x)
        return activations[-1]

    def predict(self, x):
        """Predict class label for x."""
        logger.debug(f"Predicting classes for {x.shape[0]} samples")
        proba = self.predict_proba(x)
        predictions = np.argmax(proba, axis=1)
        logger.debug(
            f"Predictions: {predictions}, "
        )
        return predictions

    def score(self, x, y):
        """Return mean accuracy on test data x and labels y."""
        logger.debug(f"Evaluating accuracy on {x.shape[0]} samples")
        
        y_true = np.argmax(y, axis=1) if y.ndim > 1 else y
        
        predictions = self.predict(x)
        
        accuracy = np.mean(predictions == y_true)
        logger.info(
            f"Accuracy: {accuracy:.4f} ({np.sum(predictions == y_true)}/{len(y_true)} correct)"
        )
        return accuracy