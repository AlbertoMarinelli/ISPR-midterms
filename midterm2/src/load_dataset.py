import numpy as np
from mlxtend.data import loadlocal_mnist


def load_data(directory): #Load data from MNIST dataset

    images_tr, labels_tr = loadlocal_mnist(images_path=directory + 'train-images-idx3-ubyte', labels_path=directory + 'train-labels-idx1-ubyte')
    images_ts, labels_ts = loadlocal_mnist(images_path=directory + 't10k-images-idx3-ubyte', labels_path=directory + 't10k-labels-idx1-ubyte')

    # Scaling training/test images and labels between 0 and 1
    min_tr = np.min(images_tr)
    min_ts = np.min(images_ts)
    max_tr = np.max(images_tr)
    max_ts = np.max(images_ts)

    images_tr = np.divide(np.subtract(images_tr, min_tr), max_tr - min_tr)
    images_ts = np.divide(np.subtract(images_ts, min_ts), max_ts - min_ts)

    return images_tr, labels_tr, images_ts, labels_ts
