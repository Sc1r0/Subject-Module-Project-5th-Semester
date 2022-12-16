# Data manipulation
import numpy as np

# Euclidean distance
from euclidean_distance import euclidean_distance_beacons as e_multi, \
    euclidean_distance_XY as e_XY
from excel_sheet_data import X_test, X, y


class KNNFromScratch:
    """
    Since there's no actual "private" instance of variables, but the naming convention suggests that variables
    that should be accessed outside the class, should have an underscore prefix. E.g. _name or _age.

    All variables will therefore adhere to this naming convention, as they should not be accesses outside of this
    class.
    """
    # initialize our object
    def __init__(self, k):
        """
        Initializes the object.
        :param k: the amount of k_nearest_neighbours, one would like to look at.
        """
        # K-value
        self._k = k
        # Data set
        self._y_train = None
        self._X_train = None
        # Needed for Prediction
        self._estimated_positions = None
        self._k_closest_ground_truths = None
        self._k_nearest_labels = None
        self._estimated_positionsXY = None
        # Needed for Prediction & Evaluation
        self._k_nearest_indexes = None
        self._all_ground_truths = None

    # fit our method
    def fit(self, X, y):
        self._X_train = X
        self._y_train = y

    # predict our estimated position(s)
    def predict(self, test_point, excel_XY_coordinates):
        # save our predictions
        self._estimated_positions = self._predict(test_point, excel_XY_coordinates)
        # calculate average of _estimated_positions, and return the singular value
        # split the array into two different list
        _estimated_positionsX, _estimated_positionsY = np.split(np.array(self._estimated_positions), 2, axis=1)
        # flatten the 2-D list to a 1-D list
        _estimated_positionsX = [item for sublist in _estimated_positionsX for item in sublist]
        _estimated_positionsY = [item for sublist in _estimated_positionsY for item in sublist]
        # calculate the average X and Y values
        _average_X = sum(_estimated_positionsX) / len(_estimated_positionsX)
        _average_Y = sum(_estimated_positionsY) / len(_estimated_positionsY)
        # add the sum of each array divided by its length to a new array - this calculates the average X and Y
        # coordinate
        self._estimated_positionsXY = [round(_average_X, 2), round(_average_Y, 2)]
        # return our predictions in a numpy array
        return self._estimated_positionsXY

    # helper method for the predict() function
    def _predict(self, test_point, excel_XY_coordinates):
        # save all ground_truths from our 'y' array containing all X,Y coordinates from collected datapoints
        self._all_ground_truths = np.array(excel_XY_coordinates)
        # Calculate euclidean distance from test_point to each index in X_train
        distances = [e_multi(test_point, x_train) for x_train in self._X_train]
        # saves the index number of the k closest neighbors (as calculated by euclidean distance)
        self._k_nearest_indexes = np.argsort(distances)[:self._k]  # returns index number of k_indices of the distances list
        # save the ground truth locations to for each index+1 in y[].
        # +1 is because lists / arrays in python starts at index 0, but our datasheet starts at datapoint 1
        self._k_closest_ground_truths = self._all_ground_truths[self._k_nearest_indexes]
        # return the closest X,Y values
        return self._k_closest_ground_truths


    def evaluate_knn_collected_rssi_values(self, prediction, test_point):
        # get predicted X,Y
        _prediction = prediction
        # get test_point GROUND TRUTH (X,Y)
        _test_point = test_point
        # fetch the index number from X where our _test_point is located
        _ground_truth_index = np.where(np.all(X.values == _test_point, axis=1))
        # fetch the ground_truth from the above index value
        _ground_truth_value = y.values[_ground_truth_index[0]]
        # calculate euclidean distance from prediction to test_point GROUND TRUTH
        distance = round(e_XY(prediction, _ground_truth_value.flatten()), 2)
        return distance


    def evaluate_knn_random_rssi_values(self, prediction):
        # save variables
        _prediction = np.array(prediction)  # our predicted X,Y coordinate
        _k_closest = self._k_closest_ground_truths  # our k_nearest ground truths as calculated in _predict()
        # calculate the distance between our prediction point to each of the ground truths in _k_closests
        _distances = [e_XY(prediction, k_ground_truth) for k_ground_truth in _k_closest]
        # calculate the average of the _k_closest euclidean distances
        _average_distance = round(sum(_distances) / len(_distances), 2)
        # return the result
        return _average_distance
