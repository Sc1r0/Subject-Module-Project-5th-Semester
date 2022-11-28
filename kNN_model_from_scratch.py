# Data manipulation
import numpy as np
import pandas as pd


# method to calculate the distance between two rows in the dataset
def euclidean_distance(x1, x2):
    distance = 0.0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i]) ** 2
    return np.sqrt(distance)


# second attempt for a method to calculate the distance between two rows in the dataset
def euclidean_distance_v2(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance


class KNNFromScratch:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # compute the distances
        distances = [euclidean_distance_v2(x, x_train) for x_train in self.X_train]

        # get closest K (indices = Latin for 'plural of index')
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
