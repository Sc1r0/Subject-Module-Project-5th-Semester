# Data manipulation
import numpy as np

# Other
from euclidean_distance import euclidean_distance, euclidean_distance_multi


class KNNFromScratch:
    # FIXME: Our predictions are wrong.
    #   Our distances formula does not calculate the euclidean distance on itself at first, and thus probably returns
    #   wrong values all the way.
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = self._predict(X)
        # print("Our predictions:", predictions)
        return np.array(predictions)

    def _predict(self, x):
        distances = [euclidean_distance_multi(x, x_train) for x_train in self.X_train]
        print("Our distances:", distances)
        # get closest K (indices = Latin for 'plural of index')
        k_indices = np.argsort(distances)[:self.k]
        # print("Our k_indices:", k_indices)
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # print("Our k_labels:", k_nearest_labels)
        return k_nearest_labels

    def margin_of_error(self, test, prediction):
        # TODO: Fill out this method
        # find euclidean distance from test_point to each index in k_indices
        # print("Our test data:", test)
        margin_of_error = euclidean_distance(test, prediction)
        # print("Our MoE:", margin_of_error)
        return margin_of_error
