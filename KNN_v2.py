# Data manipulation
import numpy as np

# Euclidean distance
from euclidean_distance import euclidean_distance as e_distance, euclidean_distance_multi as e_multi


class KNN_v2:
    """
    Since there's no actual "private" instance of variables, but the naming convention suggests that variables
    that should be accessed outside the class, should have an underscore prefix. E.g. _name or _age.

    All variables will therefore adhere to this naming convention, as they should not be accesses outside of this
    class.
    """

    # initialize our object
    def __init__(self, k):
        """
        Initialize the object
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

        # Needed for Prediction & Evaluation
        self._k_indices = None
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
        _estimated_positionsXY = [round(_average_X, 2), round(_average_Y, 2)]

        # return our predictions in a numpy array
        return _estimated_positionsXY

    # helper method for the predict() function
    def _predict(self, test_point, excel_XY_coordinates):
        # save all ground_truths from our 'y' array containing all X,Y coordinates from collected datapoints
        all_ground_truths = np.array(excel_XY_coordinates)

        # Calculate euclidean distance from test_pint to each index in X_train
        distances = [e_multi(test_point, x_train) for x_train in self._X_train]

        # saves the index number of the k closest neighbors (as calculated by euclidean distance)
        k_indices = np.argsort(distances)[:self._k]  # returns index number of k_indices of the distances list
        print("k_indices:", k_indices)

        # save the ground truth locations to for each index+1 in y[].
        # +1 is because lists / arrays in python starts at index 0, but our datasheet starts at datapoint 1
        k_closest_ground_truths = all_ground_truths[k_indices]
        print("k_closest_ground_truths:", k_closest_ground_truths)

        # return the predicted X,Y values
        return k_closest_ground_truths

    # TODO: Fill this method.
    #   Ask Xiao for guidance.
    #   issue: We have calculated the estimated X,Y position. How do we find the counter-part ground-truth to
    #   evaluate the result of our prediction?
    def evaluate_knn(self, ground_truth, prediction):
        pass

