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

    def predict(self, X, y):
        predictions = self._predict(X, y)
        # print("Our predictions:", predictions)

        return np.array(predictions)

    def _predict(self, x, y):
        global k_indices  # defined as global variable, to access it in ground_truth()
        global groundtruth  # defined as global variable, to access it in ground_truth()

        y = np.array(y)
        distances = [euclidean_distance_multi(x, x_train) for x_train in self.X_train]
        print("Our distances:", distances)
        # get closest K (indices = Latin for 'plural of index')
        k_indices = np.argsort(distances)[:self.k]  # returns index number of k_indices of the distances list
        # example: distance[66] = 39.42, which is the closest euclidean distance from the distances list.
        print("Our k_indices:", k_indices)
        print("Our k_indices:", distances[66], distances[47], distances[25], distances[67], distances[29])

        groundtruth = y[k_indices]
        print("Our groundtruths:", groundtruth)
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        return k_nearest_labels

    def root_mean_squared_error(self, predicted_value, observed_value, number_of_samples):
        # calculate the RMSE.
        # Formula: √Σ(Pi – Oi)2 / n
        # where Pi is the predicted value for the i-th observation in the dataset
        # where Oi is the observed value for the i-th observation in the dataset
        # where n is the sample size
        return np.sqrt((np.sum(predicted_value - observed_value) ** 2 / number_of_samples))


    def margin_of_error(self, test, prediction):
        # TODO: Calculate the average distance between the test-point and the true-position value.
        # find euclidean distance from test_point to each index in k_indices
        # print("Our test data:", test)
        margin_of_error = euclidean_distance(test, prediction)
        # print("Our MoE:", margin_of_error)
        return margin_of_error

    def ground_truth(self):
        groundtruthx, groundtruthy = np.split(groundtruth, 2, axis=1)

        # flatten the 2-D list to a 1-D list
        groundtruthx = [item for sublist in groundtruthx for item in sublist]
        groundtruthy = [item for sublist in groundtruthy for item in sublist]

        # calculate the average of the X and Y values
        avg_x = sum(groundtruthx) / len(groundtruthx)
        avg_y = sum(groundtruthy) / len(groundtruthy)

        # create list to save the average ground truth for the X coordinate and Y coordinate
        avg_x_y = [avg_x, avg_y]

        return avg_x_y
