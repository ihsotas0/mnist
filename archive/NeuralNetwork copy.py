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
    # if input("Load NeuralNetwork file (y/n)? ") == "y":
    #    net = load(input("File name? "))

    # Make neural network
    #net = NeuralNetwork(784, 80, 16, 10)

    # Load datasets
    train_images = load_mnist_images("dataset/train-images-idx3-ubyte")
    train_labels = load_mnist_labels("dataset/train-labels-idx1-ubyte")

    test_images = load_mnist_images("dataset/t10k-images-idx3-ubyte")
    test_labels = load_mnist_labels("dataset/t10k-labels-idx1-ubyte")

    # Format data
    train_x, train_y = format_mnist_data(train_images, train_labels)
    test_x, test_y = format_mnist_data(test_images, test_labels)

    # Train for 500 epochs or 95% accuracy (batch is entire dataset)
    stats = net.train(train_x, train_y, max_epochs=500, min_acc=95, learning_rate=1)

    # Test on entire testing dataset
    accuracy = net.test(test_x, test_y)
    print(f"Test Dataset Accuracy: {accuracy} %")

    # Save network
    unique_f = "NeuralNetwork_" + str(hash(net)) + ".pkl"
    print(f"Saved Network: {unique_f}")
    save(net, unique_f)


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


def format_mnist_data(images, labels):
    # Flatten image array
    image_list_shape = np.shape(images)
    flat_images = images.reshape(image_list_shape[0], -1)

    # Make output vectors from labels
    label_vectors = np.array([np.eye(10)[n] for n in labels])

    return flat_images, label_vectors


# Quality of life
def save(net, unique_f):
    with open(unique_f, "wb") as f:
        pickle.dump(net, f)


def load(unique_f):
    with open(unique_f, "rb") as f:
        return pickle.load(f)


## Activation functions ##


def ReLU(x):
    # Makes the function numerically stable
    x = np.clip(x, -64, 64)
    return (abs(x) + x) / 2
    # The line below was a bug I didn't notice for quite a while
    # return 1 / (1 + np.exp(x))


def d_ReLU(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


## Cost functions ##


def cross_entropy(pred, actual):
    """Categorical cross-entropy"""
    # Adding a very small epsilon to prevent log(0)
    pred = np.clip(pred, 1e-64, 1.0)
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

    def __init__(self, *layers):
        """Creates a simple feed forward neural network for categorization to be
        trained with train() and tested with test(). predict() can be used to
        run the neural network on arbitrary inputs after training for practical
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
            (rng.random((layers[i + 1], layers[i])) - 0.5) * 5  # Just works, IDK
            # The above matrix shape works for W @ x matrix multiplication
            # Rows = output size. Columns = input size
            for i in range(len(layers) - 1)
        ]

        self.biases = [
            (rng.random(layers[i]) - 0.5) * 5
            for i in range(1, len(layers))
            # Ignores input layer since it doesn't have weights or biases
        ]

        print("Weights and biases initialized randomly")

    # AI: Look over to understand
    def train(self, x, y, epochs, learning_rate):
        start_time = time()

        print("========= TRAINING BEGIN =========")

        num_samples = len(x)

        for epoch in range(epochs):

            total_cost = 0

            # Initialize EPOCH gradients FOR ALL PAIRS
            gradients_w = [np.zeros_like(w) for w in self.weights]
            gradients_b = [np.zeros_like(b) for b in self.biases]

            for xx, yy in zip(x, y):

                # Forward pass
                activations, pre_activations = self.feedforward(xx)

                # Compute cost
                total_cost += cross_entropy(activations[-1], yy)

                # Backward pass (backpropagation) for each pair
                # x is the already included in activations as the first layer of each forward pass
                grad_pair_w, grad_pair_b = self.backward(
                    activations, pre_activations, yy
                )

                # Update the gradients for specific pair in dataset
                # FIXME: I think this is why the network is so slow
                gradients_w += grad_pair_w
                gradients_b += grad_pair_b

            # Turn into average of all pairs in dataset
            # gradients_w = [gw / num_samples for gw in gradients_w]
            # gradients_b = [gb / num_samples for gb in gradients_b]
            # HACK: IDK if I'm meant to take this average. Regardless it's weird.

            # Update weights and biases
            self.update_weights_and_biases(gradients_w, gradients_b, learning_rate)

            # Print progress
            avg_cost = total_cost / num_samples
            print(f"Epoch {epoch + 1} of {epochs}")
            print(f"=> Average Cost: {avg_cost:.5f}")

            # Print accuracy every 25 epochs
            if (epoch + 1) % 25 == 0:
                accuracy = self.test(x, y)
                print(f"=> Accuracy: {accuracy:.2f}%")

        print("========= TRAINING DONE =========")

        end_time = time()
        print(f"Time taken: {(end_time - start_time) / 60:.2f} minutes")

    def backward(self, activations, pre_activations, y):
        # Initialize gradients FOR INDIVIDUAL PAIR
        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]

        # Output layer error
        d_a = d_cost(activations[-1], y)
        d_z = d_a * d_sigmoid(pre_activations[-1])

        # Update gradients for output layer
        gradients_w[-1] = np.outer(
            d_z, activations[-2]
        )  # d_z * a_(l-1)^T (the previous layer activation)
        gradients_b[-1] = d_z  # Gradient of cost w.r.t. bias is just d_z

        # Now propagate back to hidden layers
        for l in range(2, len(self.layers)):  # Start from second to last layer
            # Compute the gradient for the hidden layer
            d_a = self.weights[-l + 1].T @ d_z  # Backpropagate error
            d_z = d_a * d_sigmoid(pre_activations[-l])  # Apply sigmoid derivative

            # Update gradients for this layer
            gradients_w[-l] = np.outer(
                d_z, activations[-l - 1]
            )  # d_z * a_(l-1)^T (the previous layer activation)
            gradients_b[-l] = d_z  # Gradient of cost w.r.t. bias is just d_z

        return gradients_w, gradients_b

    def update_weights_and_biases(self, gradients_w, gradients_b, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients_w[i]
            self.biases[i] -= learning_rate * gradients_b[i]

    def feedforward(self, x):
        # List to store activations at each layer
        activations = [x]
        # List to store pre-activations (inputs to the activation function)
        pre_activations = []

        layer = x  # Input layer

        # Iterate through each layer of the network
        for w, b in zip(self.weights, self.biases):
            pre_activation = w @ layer + b  # Perform matrix multiplication
            layer = sigmoid(pre_activation)  # Apply sigmoid activation function

            activations.append(layer)  # Store activations
            pre_activations.append(pre_activation)  # Store pre-activations

        activations[-1] = softmax(activations[-1])

        return activations, pre_activations

    def test(self, x, y):
        num_correct, num_wrong = 0, 0

        for xx, yy in zip(x, y):
            guess, _ = self.feedforward(xx)
            if np.argmax(guess[-1]) == np.argmax(yy):
                num_correct += 1
            else:
                num_wrong += 1

        # Return accuracy percent
        return 100 * (num_correct / (num_correct + num_wrong))

    def predict(self, x):
        pred, _ = self.feedforward(x)
        return np.argmax(pred[-1])


if __name__ == "__main__":
    main()
