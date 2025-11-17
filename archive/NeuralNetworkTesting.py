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
    stats = net.train(train_x, train_y, epochs=5000, batch_size=0, learning_rate=0.001)  # Reduced learning rate

    # Test on entire testing dataset
    accuracy = net.test(test_x, test_y)

    # Save network
    unique_f = "NeuralNetwork_" + str(hash(net)) + ".pkl"
    save(net, unique_f)

    print(f"Test Dataset Accuracy: {accuracy:.2f}%")

    # For user testing
    playground()

## Post-training playground ##
def playground():
    pass

## Data functions ##
def load_mnist_images(filename):
    with open(filename, "rb") as f:
        magic, num_items, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, rows, cols)
    return images / 255  # Normalizes uint8 to float in [0,1]

def load_mnist_labels(filename):
    with open(filename, "rb") as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def one_hot(x, num_classes):
    return np.eye(num_classes)[x]

def format_mnist_data(images, labels):
    image_list_shape = np.shape(images)
    flat_images = images.reshape(image_list_shape[0], -1)

    label_vectors = np.array([one_hot(n, 10) for n in labels])

    return flat_images, label_vectors

# Quality of life
def save(net, unique_f):
    with open(unique_f, "wb") as f:
        pickle.dump(net, f)

def load(unique_f):
    with open(unique_f, "rb") as f:
        return pickle.load(f)

## Activation functions ##
def sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

## Cost functions ##
def cross_entropy(pred, actual):
    pred = np.clip(pred, 1e-15, 1.0 - 1e-15)
    return -np.sum(actual * np.log(pred), axis=1)

def d_cost(pred, actual):
    return pred - actual

## Neural Network ##
class NeuralNetwork:
    def __init__(self, layers):
        print(f"New neural network: {layers}")

        self.layers = layers

        # Xavier initialization for weights
        self.weights = [
            rng.randn(layers[i + 1], layers[i]) / np.sqrt(layers[i])
            for i in range(len(layers) - 1)
        ]

        self.biases = [
            rng.randn(layers[i]) * 0.01
            for i in range(1, len(layers))
        ]

        print("Weights and biases initialized with Xavier initialization")

    def train(self, x, y, epochs, batch_size, learning_rate):
        start_time = time()

        print("========= TRAINING BEGIN =========")

        num_samples = len(x)

        if batch_size == 0:
            batch_size = num_samples

        num_batches = num_samples // batch_size if batch_size > 0 else 1
        for epoch in range(epochs):
            if batch_size > 0:
                indices = rng.permutation(num_samples)
                x_shuffled = x[indices]
                y_shuffled = y[indices]
            else:
                x_shuffled = x
                y_shuffled = y

            epoch_cost = 0
            for batch in range(num_batches):
                start = batch * batch_size
                end = (batch + 1) * batch_size
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                activations, pre_activations = self.forward(x_batch)

                batch_cost = np.mean(cross_entropy(activations[-1], y_batch))
                epoch_cost += batch_cost

                gradients_w, gradients_b = self.backward(x_batch, y_batch, activations, pre_activations)

                self.update_weights_and_biases(gradients_w, gradients_b, learning_rate)

            avg_cost = epoch_cost / num_batches
            print(f"Epoch {epoch + 1}/{epochs} | Cost: {avg_cost:.5f}")

            if (epoch + 1) % 25 == 0:
                accuracy = self.test(x, y)
                print(f"=> Accuracy: {accuracy:.2f}%")

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

        return 100 * (num_correct / (num_correct + num_wrong))

    def feedforward(self, x):
        layer = x

        for w, b in zip(self.weights, self.biases):
            layer = sigmoid(w @ layer + b)

        return softmax(layer)

    def backward(self, x_batch, y_batch, activations, pre_activations):
        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]

        d_a = d_cost(activations[-1], y_batch)
        d_z = d_a * d_sigmoid(pre_activations[-1])

        gradients_w[-1] = d_z.T @ activations[-2]
        gradients_b[-1] = np.sum(d_z, axis=0)

        for l in range(2, len(self.layers)):
            d_a = d_z @ self.weights[-l + 1]
            d_z = d_a * d_sigmoid(pre_activations[-l])

            gradients_w[-l] = d_z.T @ activations[-l - 1]
            gradients_b[-l] = np.sum(d_z, axis=0)

        return gradients_w, gradients_b

    def update_weights_and_biases(self, gradients_w, gradients_b, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients_w[i]
            self.biases[i] -= learning_rate * gradients_b[i]

    def forward(self, x):
        activations = [x]
        pre_activations = []

        layer = x

        for w, b in zip(self.weights, self.biases):
            pre_activation = np.dot(layer, w.T) + b
            layer = sigmoid(pre_activation)

            activations.append(layer)
            pre_activations.append(pre_activation)

        return activations, pre_activations

    def predict(self, x):
        pred = self.feedforward(x)
        return np.argmax(pred)

if __name__ == "__main__":
    main()