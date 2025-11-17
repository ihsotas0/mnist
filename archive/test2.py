import pickle
import struct
from math import *

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng
import psutil


# Helper functions
def _format_data(images, labels):
    # Flatten image array
    image_list_shape = np.shape(images)
    flat_images = images.reshape(image_list_shape[0], -1)

    # Make output vectors from labels
    label_list_shape = np.shape(labels)
    label_scope = len(set(labels))  # HACK, too slow?
    label_vectors = np.zeros((label_list_shape[0], label_scope))

    # Create "hot ones" in output vectors
    for i, n in enumerate(labels):
        label_vectors[i, n] = 1.0

    return flat_images, label_vectors


# Load files from https://yann.lecun.com/exdb/mnist/
def load_mnist_images(filename):
    with open(filename, "rb") as f:
        # Read the magic number and number of items
        magic, num_items, rows, cols = struct.unpack(">IIII", f.read(16))
        # Read the image data as a numpy array
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, rows, cols)
    return images / 256  # Normalizes uint8 to float in [0,1]


def load_mnist_labels(filename):
    with open(filename, "rb") as f:
        # Read the magic number and number of items
        magic, num_items = struct.unpack(">II", f.read(8))
        # Read the label data as a numpy array
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


# MSE loss function and gradient
def _mse_loss(predictions, actual):
    return np.mean((predictions - actual) ** 2)


def _d_mse_loss(predictions, actual):
    return 2 * (predictions - actual)


# Sigmoid function and its derivative
def _sigmoid(x):
    return x * (x > 0)
    # nums = np.clip(nums, -10, 10)
    # return 1 / (1 + np.exp(nums))


def _d_sigmoid(x):
    return 1.0 * (x > 0)
    # sig = _sigmoid(nums)
    # return sig * (1 - sig)


class NeuralNetwork:

    def __init__(self, params):
        self.hyperparameters = params
        print(f"New neural network with parameters: {params}")

        # Xavier Initialization
        self.weights = [
            np.random.randn(params[i + 1], params[i])
            * np.sqrt(2.0 / (params[i] + params[i + 1]))
            for i in range(len(params) - 1)
        ]
        self.biases = [rng.random(params[i]) for i in range(1, len(params))]
        print("Weights and biases initialized using Xavier initialization")

    def train(self, x, y, epochs, batch_size=0, learn_rate=0.01):
        n_samples = x.shape[0]
        n_layers = len(self.weights)

        for epoch in range(epochs):
            epoch_cost = 0
            if batch_size > 0:
                for i in range(0, n_samples, batch_size):
                    x_batch, y_batch = x[i : i + batch_size], y[i : i + batch_size]
                    batch_cost = self._update_weights(x_batch, y_batch, learn_rate)
                    epoch_cost += batch_cost
                    print(f"-> Batch {i + 1}, Batch Cost: {batch_cost}")
            else:
                epoch_cost = self._update_weights(x, y, learn_rate)

            # Average cost for the epoch
            epoch_cost /= n_samples
            print(f"Epoch {epoch + 1}, Avg. Cost: {100*epoch_cost}")

        return self.weights, self.biases

    def _update_weights(self, x, y, learn_rate):
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(activations[-1], w.T) + b
            zs.append(z)
            activations.append(_sigmoid(z))

        cost = _mse_loss(activations[-1], y)
        delta = _d_mse_loss(activations[-1], y) * _d_sigmoid(zs[-1])

        # Apply gradient clipping
        grad_clip = 1.0  # Clip gradients to this value
        self.weights[-1] -= learn_rate * np.clip(
            np.dot(delta.T, activations[-2]), -grad_clip, grad_clip
        )
        self.biases[-1] -= learn_rate * np.clip(
            np.sum(delta, axis=0), -grad_clip, grad_clip
        )

        for l in range(2, len(self.weights) + 1):
            delta = np.dot(delta, self.weights[-l + 1]) * _d_sigmoid(zs[-l])
            self.weights[-l] -= learn_rate * np.clip(
                np.dot(delta.T, activations[-l - 1]), -grad_clip, grad_clip
            )
            self.biases[-l] -= learn_rate * np.clip(
                np.sum(delta, axis=0), -grad_clip, grad_clip
            )

        return cost

    def im_train(self, images, labels, epochs, batch_size=0, learn_rate=0.01):
        flat_images, label_vectors = _format_data(images, labels)
        w, b = self.train(flat_images, label_vectors, epochs, batch_size, learn_rate)
        return w, b

    def im_predict(self, images, labels):
        flat_images, label_vectors = _format_data(images, labels)
        num_correct = 0
        for image, label in zip(flat_images, label_vectors):
            guess = self._feedforward(image)
            if np.argmax(guess) == np.argmax(label):
                num_correct += 1
        return 100 * num_correct / len(labels)

    def _feedforward(self, x):
        layer = x
        for w, b in zip(self.weights, self.biases):
            layer = _sigmoid(np.dot(layer, w.T) + b)
        return layer

    def save(self):  # Quality of life
        with open("neural_network.pkl", "wb") as outp:
            pickle.dump(self.hyperparameters, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.weights, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.biases, outp, pickle.HIGHEST_PROTOCOL)


# Main training
def main():
    # Initialize the neural network
    net = NeuralNetwork([784, 16, 16, 10])

    # Load MNIST data
    train_images = load_mnist_images("MNIST/train-images-idx3-ubyte")
    train_labels = load_mnist_labels("MNIST/train-labels-idx1-ubyte")
    test_images = load_mnist_images("MNIST/t10k-images-idx3-ubyte")
    test_labels = load_mnist_labels("MNIST/t10k-labels-idx1-ubyte")

    # Train the network
    w, b = net.im_train(train_images, train_labels, epochs=2000, learn_rate=0.1)

    # Evaluate accuracy on the test set
    accuracy = net.im_predict(test_images, test_labels)
    print(f"Accuracy: {accuracy}%")

    net.save()


if __name__ == "__main__":
    main()
