import pickle
import struct

import numpy as np


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

# TODO: Add hashing for filenames and timestamps

def save(net, unique_f):
    with open(unique_f, "wb") as f:
        pickle.dump(net, f)


def load(unique_f):
    with open(unique_f, "rb") as f:
        return pickle.load(f)