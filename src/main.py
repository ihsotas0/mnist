import logging
import sys
import time

import matplotlib.pyplot as plt

from file_tools import *
from neural_network import NeuralNetwork

logger = logging.getLogger(__name__)


def main():
    argv = sys.argv

    timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
    log_file = f"../logs/mnist_{timestr}.log"

    #fh = logging.FileHandler(f"../logs/mnist_{timestr}.log")
    #ch = logging.StreamHandler()

    # HACK: Implement better system
    if "-v" in argv:
        logger.basicConfig(level=logging.DEBUG)
        #logger.setLevel(logging.DEBUG)
        #ch.setLevel(logging.DEBUG)
        #fh.setLevel(logging.DEBUG)
    else:
        logger.basicConfig(level=logging.INFO)
        #logger.setLevel(logging.INFO)
        #ch.setLevel(logging.INFO)
        #fh.setLevel(logging.INFO)
    if "-l" in argv:
        net = load(input("Network name? "))
    else:
        net = NeuralNetwork(max_iter=20)

    #logger.addHandler(ch)
    #logger.addHandler(fh)

    logger.debug("Initialized logging")

    # Load datasets
    train_images = load_mnist_images("../data/train-images-idx3-ubyte")
    train_labels = load_mnist_labels("../data/train-labels-idx1-ubyte")

    test_images = load_mnist_images("../data/t10k-images-idx3-ubyte")
    test_labels = load_mnist_labels("../data/t10k-labels-idx1-ubyte")

    # Format data
    x_train, y_train = format_mnist_data(train_images, train_labels)
    x_val, y_val = format_mnist_data(test_images, test_labels)

    net.fit(x_train, y_train, validation_data=(x_val, y_val))

    plt.plot(net.loss_curve, label="Loss")
    plt.plot(net.validation_scores, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.show()

    save(net, f"../models/mnist_net_{timestr}.pkl")


if __name__ == "__main__":
    main()
