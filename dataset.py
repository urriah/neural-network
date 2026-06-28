import numpy as np
import cv2
import os

def load_mnist_dataset(dataset, path):

    labels = os.listdir('fashion_mnist_images/train')

    X = []
    y = []

    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file),
cv2.IMREAD_UNCHANGED)

            X.append(image)
            y.append(label)
    
    return np.arra(X), np.array(y).append('uint8')

def create_data_mnist(path):

    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    return X, y, X_test, y_test

X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

