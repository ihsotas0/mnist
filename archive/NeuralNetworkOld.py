import pickle
import struct
from math import *
from time import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng

## Implementation of a NeuralNetwork for MNIST dataset classification ##


def main():
    # Check for pre-existing network
    if input("Load NeuralNetwork file (y/n)? ") == "y":
        net = load(input("File name? "))

    # Make neural network
    net = NeuralNetwork([784, 80, 16, 10])

    # Load datasets
    train_images = load_mnist_images("dataset/train-images-idx3-ubyte")
    train_labels = load_mnist_labels("dataset/train-labels-idx1-ubyte")

    test_images = load_mnist_images("dataset/t10k-images-idx3-ubyte")
    test_labels = load_mnist_labels("dataset/t10k-labels-idx1-ubyte")

    # Format data
    train_x, train_y = format_mnist_data(train_images, train_labels)
    test_x, test_y = format_mnist_data(test_images, test_labels)

    # Train for 5000 epochs with batch size of 0 (entire dataset)
    stats = net.train(train_x, train_y, epochs=5000, batch_size=0, learning_rate=0.01)

    # Test on entire testing dataset
    accuracy = net.test(test_x, test_y)

    # Save network
    unique_f = "NeuralNetwork_" + str(hash(net)) + ".pkl"
    save(net, unique_f)

    print(f"Test Dataset Accuracy: {accuracy} %")

    # For user testing
    playground()


## Post-training playground ##
def playground():

    # ChatGPT, ignore this for now.

    pass


## Data functions ##


# Load files from https://yann.lecun.com/exdb/mnist/
def load_mnist_images(filename):
    with open(filename, "rb") as f:
        # Read the magic number and number of items
        magic, num_items, rows, cols = struct.unpack(">IIII", f.read(16))
        # Read the image data as a numpy array
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, rows, cols)
    return images / 255  # Normalizes uint8 to float in [0,1]


def load_mnist_labels(filename):
    with open(filename, "rb") as f:
        # Read the magic number and number of items
        magic, num_items = struct.unpack(">II", f.read(8))
        # Read the label data as a numpy array
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def one_hot(x, num_classes):
    """Create a one-hot vector given num_classes and the class num, x."""
    return np.eye(num_classes)[x]


def format_mnist_data(images, labels):
    # Flatten image array
    image_list_shape = np.shape(images)
    flat_images = images.reshape(image_list_shape[0], -1)

    # Make output vectors from labels
    label_list_shape = np.shape(labels)
    num_classes = len(set(labels))  # HACK, too slow?
    label_vectors = np.array([one_hot(n, num_classes) for n in labels])

    return flat_images, label_vectors


# Quality of life
def save(net, unique_f):
    with open(unique_f, "wb") as f:
        pickle.dump(net, f)


def load(unique_f):
    """Returns object"""
    with open(unique_f, "rb") as f:
        return pickle.load(f)


## Activation functions ##


def sigmoid(x):
    # Makes the function numerically stable
    x = np.clip(x, -10, 10)
    return 1 / (1 + np.exp(-x))
    # The line below was a bug I didn't notice for quite a while
    # return 1 / (1 + np.exp(x))


def d_sigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


def softmax(z):
    exp_z = np.exp(z - np.max(z))  # Numerical stability
    return exp_z / np.sum(exp_z)


## Cost functions ##


def cross_entropy(pred, actual):
    """Categorical cross-entropy"""
    pred = np.clip(
        pred, 1e-15, 1.0 - 1e-15
    )  # Adding a very small epsilon to prevent log(0)
    return -np.sum(actual * np.log(pred))  # Cross-entropy loss


def d_cost(pred, actual):
    """Derivative of categorical cross-entropy and softmax together"""
    return pred - actual


## Neural Network ##


class NeuralNetwork:
    """
    The NeuralNetwork class can be used to make simple feedforward
    classification neural networks with custom hyperparameters and a sigmoid
    activation function.
    """

    def __init__(self, layers):
        """Creates a simple feed forward neural network for categorization to be
        trained with train() and tested with test(). predict() can be used to
        run the neural network on arbitary inputs after training for practical
        use.

        Args:
            layers (list[int]): The number of neurons in each fully connected
            layer, starting with the input vector and ending with the output
            vector.
        """
        print(f"New neural network: {layers}")

        self.layers = layers

        # Initialize random weights and biases for training
        self.weights = [
            rng.random((layers[i + 1], layers[i]))
            # The above matrix shape works for W @ x matrix multiplication
            # Rows = output size. Columns = input size
            for i in range(len(layers) - 1)
        ]

        self.biases = [
            # Small initial biases
            rng.random(layers[i]) * 0.01
            for i in range(1, len(layers))
            # Ignores input layer since it doesn't have weights or biases
        ]

        print("Weights and biases initialized randomly")

    # AI: Look over to understand
    def train(self, x, y, epochs, batch_size, learning_rate):
        start_time = time()

        print("========= TRAINING BEGIN =========")

        num_samples = len(x)

        # Handle case when batch_size is 0
        if batch_size == 0:
            batch_size = num_samples  # Use the entire dataset as one batch

        num_batches = num_samples // batch_size if batch_size > 0 else 1
        for epoch in range(epochs):
            # Shuffle data if batch_size > 0
            if batch_size > 0:
                indices = rng.permutation(num_samples)
                x_shuffled = x[indices]
                y_shuffled = y[indices]
            else:
                x_shuffled = x
                y_shuffled = y

            epoch_cost = 0
            for batch in range(num_batches):
                if batch_size > 0:
                    start = batch * batch_size
                    end = (batch + 1) * batch_size
                    x_batch = x_shuffled[start:end]
                    y_batch = y_shuffled[start:end]
                else:
                    # When batch_size = 0, treat the entire dataset as a single batch
                    x_batch = x_shuffled
                    y_batch = y_shuffled

                # Forward pass
                activations, pre_activations = self.forward(x_batch)

                # Compute cost for this batch
                batch_cost = np.mean(
                    [
                        cross_entropy(activation, actual)
                        for activation, actual in zip(activations[-1], y_batch)
                    ]
                )
                epoch_cost += batch_cost

                # Backward pass (backpropagation)
                gradients_w, gradients_b = self.backward(
                    x_batch, y_batch, activations, pre_activations
                )

                # Update weights and biases
                self.update_weights_and_biases(gradients_w, gradients_b, learning_rate)

            # Print progress
            avg_cost = epoch_cost / num_batches
            print(f"Epoch {epoch + 1}/{epochs} | Cost: {avg_cost:.5f}")

            # Print accuracy every 25 epochs
            if (epoch + 1) % 25 == 0:
                accuracy = self.test(x, y)
                print(f"    => Accuracy: {accuracy:.2f}% <=")

        print("========= TRAINING DONE =========")

        end_time = time()
        print(f"Time taken: {(end_time - start_time) / 60:.2f} minutes")

    def test(self, x, y):
        num_correct, num_wrong = 0, 0

        for xx, yy in zip(x, y):
            guess = self.feedforward(xx)
            if np.argmax(guess) == np.argmax(yy):
                num_correct += 1
            else:
                num_wrong += 1

        # Return accuracy percent
        return 100 * (num_correct / (num_correct + num_wrong))

    def feedforward(self, x):
        layer = x

        for w, b in zip(self.weights, self.biases):
            layer = sigmoid(w @ layer + b)

        return softmax(layer)

    # AI: Look over to understand
    def backward(self, x_batch, y_batch, activations, pre_activations):
        # Initialize gradients
        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]

        # Output layer error
        d_a = d_cost(activations[-1], y_batch)
        d_z = d_a * d_sigmoid(pre_activations[-1])

        gradients_w[-1] = d_z.T @ activations[-2]
        gradients_b[-1] = np.sum(d_z, axis=0)

        # Hidden layers error (backpropagate)
        for l in range(2, len(self.layers)):
            d_a = d_z @ self.weights[-l + 1]
            d_z = d_a * d_sigmoid(pre_activations[-l])

            gradients_w[-l] = d_z.T @ activations[-l - 1]
            gradients_b[-l] = np.sum(d_z, axis=0)

        return gradients_w, gradients_b

    # AI: Look over to understand
    def update_weights_and_biases(self, gradients_w, gradients_b, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients_w[i]
            self.biases[i] -= learning_rate * gradients_b[i]

    # AI: Look over to understand
    def forward(self, x):
        """
        Perform the forward pass.
        x: numpy array of shape (batch_size, input_size) where input_size = 784 for MNIST
        """
        activations = [x]  # List to store activations at each layer
        pre_activations = (
            []
        )  # List to store pre-activations (inputs to the activation function)

        layer = x  # Input layer

        # Iterate through each layer of the network
        for w, b in zip(self.weights, self.biases):
            pre_activation = np.dot(layer, w.T) + b  # Perform matrix multiplication
            layer = sigmoid(pre_activation)  # Apply sigmoid activation function

            activations.append(layer)  # Store activations
            pre_activations.append(pre_activation)  # Store pre-activations

        return activations, pre_activations

    def predict(self, x):
        pred = self.feedforward(x)
        return np.argmax(pred)  # HACK assumes categories are [0,n]


if __name__ == "__main__":
    main()


## Broken AI train() snippets ##


# num_samples = len(x)

# for epoch in range(epochs):
#     permutation = rng.permutation(num_samples)
#     x_shuffled = x[permutation]
#     y_shuffled = y[permutation]

#     total_cost = 0

#     for i in range(
#         0, num_samples, batch_size if batch_size > 0 else num_samples
#     ):
#         x_batch = (
#             x_shuffled[i : i + batch_size] if batch_size > 0 else x_shuffled
#         )
#         y_batch = (
#             y_shuffled[i : i + batch_size] if batch_size > 0 else y_shuffled
#         )

#         activations = [x_batch]
#         zs = []

#         for w, b in zip(self.weights, self.biases):
#             z = activations[-1] @ w.T + b
#             zs.append(z)
#             activations.append(self.activate_func(z))

#         output = softmax(activations[-1])
#         cost_gradient = d_loss(output, y_batch)

#         deltas = [cost_gradient * self.d_activate_func(zs[-1])]

#         for l in range(len(self.weights) - 2, -1, -1):
#             delta = (deltas[-1] @ self.weights[l + 1]) * self.d_activate_func(
#                 zs[l]
#             )
#             deltas.append(delta)

#         deltas.reverse()

#         for l in range(len(self.weights)):
#             grad_w = deltas[l].T @ activations[l]
#             grad_b = np.sum(deltas[l], axis=0)

#             # Clip gradients to avoid exploding gradients
#             grad_w = np.clip(grad_w, -clip_value, clip_value)
#             grad_b = np.clip(grad_b, -clip_value, clip_value)

#             # Update weights and biases
#             self.weights[l] -= learn_rate * grad_w
#             self.biases[l] -= learn_rate * grad_b

#         total_cost += np.sum(cross_entropy(output, y_batch))

# def _update_weights(self, x, y, learn_rate):
#     # Forward pass: calculate activations
#     activations = [x]
#     zs = []  # List to store z vectors (weighted inputs to each layer)
#     for w, b in zip(self.weights, self.biases):
#         z = np.dot(activations[-1], w.T) + b
#         zs.append(z)
#         activations.append(_sigmoid(z))

#     # Backward pass: compute gradients and update weights and biases
#     cost = _mse_loss(activations[-1], y)

#     # Compute the error delta for the output layer
#     delta = _d_mse_loss(activations[-1], y) * _d_sigmoid(zs[-1])

#     # Update weights and biases for the output layer
#     self.weights[-1] -= learn_rate * np.dot(delta.T, activations[-2])
#     self.biases[-1] -= learn_rate * np.sum(delta, axis=0)

#     # Update weights and biases for the hidden layers
#     for l in range(2, len(self.weights) + 1):
#         delta = np.dot(delta, self.weights[-l + 1]) * _d_sigmoid(zs[-l])
#         self.weights[-l] -= learn_rate * np.dot(delta.T, activations[-l - 1])
#         self.biases[-l] -= learn_rate * np.sum(delta, axis=0)

#     return np.sum(cost)

# n_samples = x.shape[0]
# n_layers = len(self.weights)

# # Loop through epochs
# for epoch in range(epochs):
#     print(f"Begin epoch {epoch + 1}")

#     if batch_size == 0:
#         avg_cost = self.update_weights(x, y, learn_rate)
#     # Mini-batch gradient descent
#     else:
#         for i in range(0, n_samples, batch_size):
#             x_batch, y_batch = x[i : i + batch_size], y[i : i + batch_size]
#             avg_cost = self._update_weights(x_batch, y_batch, learn_rate)
#             print(f"-> Batch {i + 1}, Cost: {avg_cost}")

#     print


# Number of samples in the dataset
# num_samples = len(x)

# for epoch in range(epochs):
#     # Shuffle data to ensure different order every epoch
#     permutation = rng.permutation(num_samples)
#     x_shuffled = x[permutation]
#     y_shuffled = y[permutation]

#     for i in range(0, num_samples, batch_size):
#         # Batch selection
#         batch_x = x_shuffled[i : i + batch_size]
#         batch_y = y_shuffled[i : i + batch_size]

#         # Perform forward pass for the entire batch
#         activations = [batch_x]  # List of activations for each layer
#         z_values = []  # List of weighted inputs to each layer

#         # Forward pass
#         for w, b in zip(self.weights, self.biases):
#             z = batch_x @ w.T + b
#             z_values.append(z)
#             batch_x = sigmoid(z)  # Activation after sigmoid
#             activations.append(batch_x)

#         # Compute the loss and its gradient
#         output = activations[-1]
#         cost_gradient = d_cost(output, batch_y)  # Gradient of the cost function

#         # Backpropagation
#         # Gradients for weights and biases
#         deltas = [cost_gradient * d_sigmoid(z_values[-1])]  # Output layer delta

#         # Iterate backwards through the layers
#         for l in range(len(self.weights) - 2, -1, -1):
#             delta = deltas[-1] @ self.weights[l + 1] * d_sigmoid(z_values[l])
#             deltas.append(delta)

#         deltas.reverse()  # Reverse deltas to correspond with each layer

#         # Update weights and biases using the gradients
#         for l in range(len(self.weights)):
#             # Gradient for weights and biases
#             grad_w = deltas[l].T @ activations[l]
#             grad_b = np.sum(deltas[l], axis=0)

#             # Update weights and biases using gradient descent
#             self.weights[l] -= learn_rate * grad_w
#             self.biases[l] -= learn_rate * grad_b

#     # Print accuracy every 5 epochs
#     if (epoch + 1) % 5 == 0:
#         accuracy = self.test(x, y)
#         print(f"Epoch {epoch + 1}/{epochs} - Accuracy: {accuracy:.2f}%")


# for epoch in range(epochs):
#     # Split into batches
#     for batch in range(0, len(x), batch_size):
#         x_batch = x[batch:batch + batch_size]
#         y_batch = y[batch:batch + batch_size]

#         # Forward pass
#         layer_values = [x_batch]
#         for i, (w, b) in enumerate(zip(self.weights, self.biases)):
#             layer_values.append(sigmoid(np.dot(layer_values[-1], w.T) + b))

#         # Backward pass
#         dw = [None] * len(self.weights)
#         db = [None] * len(self.biases)
#         d_layer_values = [None] * len(layer_values)

#         # Calculate output error
#         d_layer_values[-1] = d_cost(softmax(layer_values[-1]), y_batch)

#         # Backpropagate error
#         for i in range(len(self.weights) - 1, -1, -1):
#             d_layer_values[i] = np.dot(d_layer_values[i + 1], self.weights[i]) * d_sigmoid(layer_values[i])
#             dw[i] = np.dot(d_layer_values[i + 1].T, layer_values[i])
#             db[i] = np.sum(d_layer_values[i + 1], axis=0)

#         # Weight updates
#         for i in range(len(self.weights)):
#             self.weights[i] -= learn_rate * dw[i]
#             self.biases[i] -= learn_rate * db[i]

#     # Print loss every 10 epochs
#     if (epoch + 1) % 10 == 0:
#         acc = self.test(x, y)
#         print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {acc} %")


# def train(self, x, y, epochs=2000, batch_size=0, learn_rate=0.05):
#     """Trains the neural network using backpropagation on input and output
#     vectors for a number of epochs with certain batch sizes. A batch size of
#     0 means the entire training set is used for each gradient descent.

#     Args:
#         x (ndarray): Input dataset (list of vectors).
#         y (ndarray): Output dataset (list of vectors).
#     """

#     start_time = time()
#     print("========= TRAINING BEGIN =========")

#     num_samples = len(x)

#     for epoch in range(epochs):
#         permutation = rng.permutation(num_samples)
#         x_shuffled = x[permutation]
#         y_shuffled = y[permutation]

#         total_cost = 0

#         for i in range(
#             0, num_samples, batch_size if batch_size > 0 else num_samples
#         ):
#             x_batch = (
#                 x_shuffled[i : i + batch_size] if batch_size > 0 else x_shuffled
#             )
#             y_batch = (
#                 y_shuffled[i : i + batch_size] if batch_size > 0 else y_shuffled
#             )

#             activations = [x_batch]
#             zs = []

#             for w, b in zip(self.weights, self.biases):
#                 z = activations[-1] @ w.T + b
#                 zs.append(z)
#                 activations.append(self.activate_func(z))

#             output = activations[-1]
#             cost_gradient = d_loss(output, y_batch)

#             deltas = [cost_gradient * self.d_activate_func(zs[-1])]

#             for l in range(len(self.weights) - 2, -1, -1):
#                 delta = (deltas[-1] @ self.weights[l + 1]) * self.d_activate_func(
#                     zs[l]
#                 )
#                 deltas.append(delta)

#             deltas.reverse()

#             for l in range(len(self.weights)):
#                 grad_w = deltas[l].T @ activations[l]
#                 grad_b = np.sum(deltas[l], axis=0)

#                 # Debugging prints
#                 print(f"Layer {l} grad_w: {np.mean(grad_w)}")
#                 print(f"Layer {l} grad_b: {np.mean(grad_b)}")

#                 # Gradient checks
#                 if np.any(np.isnan(grad_w)) or np.any(np.isnan(grad_b)):
#                     print(f"NaN detected in gradients at layer {l}")

#                 grad_w = np.clip(grad_w, -1e5, 1e5)
#                 grad_b = np.clip(grad_b, -1e5, 1e5)

#                 self.weights[l] -= learn_rate * grad_w
#                 self.biases[l] -= learn_rate * grad_b

#             total_cost += np.sum(cross_entropy(output, y_batch))

#         if (epoch + 1) % 10 == 0:
#             avg_cost = total_cost / num_samples
#             accuracy = self.test(x, y)
#             print(
#                 f"Epoch {epoch + 1}/{epochs} - Cost: {avg_cost:.5f} - Accuracy: {accuracy:.2f}%"
#             )

#     print("========= TRAINING DONE =========")
#     end_time = time()
#     print(f"Time taken: {(end_time - start_time) / 60} minutes")

#     return self.weights, self.biases

#     print("========= TRAINING DONE =========")
#     end_time = time()

#     # Print stats
#     print(f"Time taken: {(end_time - start_time) / 60} minutes")

#     return self.weights, self.biases
