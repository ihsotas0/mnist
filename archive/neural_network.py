import struct
from math import *

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng
import psutil


# Load files from https://yann.lecun.com/exdb/mnist/
def load_mnist_images(filename):
    with open(filename, "rb") as f:
        # Read the magic number and number of items
        magic, num_items, rows, cols = struct.unpack(">IIII", f.read(16))
        # Read the image data as a numpy array
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, rows, cols)
    return images / 256


def load_mnist_labels(filename):
    with open(filename, "rb") as f:
        # Read the magic number and number of items
        magic, num_items = struct.unpack(">II", f.read(8))
        # Read the label data as a numpy array
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


# Feedforward network functions
def feedforward(image, weights, biases):
    layer = image.flatten()

    for w, b in zip(weights, biases):
        layer = sigmoid(w @ layer + b)

    return softmax(layer)


def compute_loss(pred, actual):
    # Make array of desired output
    a = np.zeros(np.size(pred))
    a[actual - 1] = 1.0

    # MSE loss function
    return np.sum((a - pred) ** 2) / np.size(pred)


# Helper functions
def sigmoid(nums):
    return 1 / (1 + np.exp(nums))


def softmax(nums, beta=1.0):
    expons = np.exp(beta * nums)
    return expons / np.sum(expons)


def main():
    # Load training data
    train_images = load_mnist_images("MNIST/train-images-idx3-ubyte")
    train_labels = load_mnist_labels("MNIST/train-labels-idx1-ubyte")

    # List of layer sizes in feedforward network, including input and output
    params = [784, 1568, 1568, 784, 28, 10]

    # For matrix multiplication, since activations are row vectors (and W @ X
    # order of operations), the number of columns in the weights matrix is the
    # input and the number of rows is the output (the activations of the layer)
    weights = [
        rng.random((params[i + 1], params[i])) - 0.5 for i in range(len(params) - 1)
    ]
    biases = [rng.random(params[i]) - 0.5 for i in range(1, len(params))]

    # Training
    for i, (img, label) in enumerate(zip(train_images, train_labels)):

        plt.title(f"Training... [epoch={i}]")
        plt.imshow(img, cmap="gray")
        plt.xlabel(f"Label: {label}")
        prediction = feedforward(img, weights, biases)
        plt.ylabel(f"Prediction: {np.argmax(prediction) + 1}")
        plt.draw()
        plt.pause(1)
        plt.clf()

    # Testing

    # Load testing data
    test_images = load_mnist_images("MNIST/t10k-images-idx3-ubyte")
    test_labels = load_mnist_labels("MNIST/t10k-labels-idx1-ubyte")

    for img, label in zip(test_images, test_labels):
        pass


if __name__ == "__main__":
    main()
