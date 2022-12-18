# KNN Libraries
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

# Our data
from excel_sheet_data import X, X_train, y, y_train, y_test


class KNNWithLibraries:
    """
    This class contains the KNN method developed through the use of the NumPy and Sci-Kit Learn libraries.
    """
    def __init__(self, k):
        """
        Initializes the object.
        :param k: the amount of k_nearest_neighbours, one would like to look at.
        """
        self.KNN_model = KNeighborsRegressor(n_neighbors=k)
        self.KNN_predictions = None
        self.KNN_eval = None

    def fit(self):
        """
        Fits the model with the X_train and y_train datasets.
        """
        # instance and fit the model
        self.KNN_model.fit(X_train.values, y_train.values)

    def predict(self, test_point):
        """
        Predicts the X,Y location.
        :return: The predicted X,Y location
        """
        # save test_point variable and reshape multi-dimensional array into a 1-D array
        test_point = np.array(test_point)
        test_point = test_point.reshape(1, -1)
        self.KNN_predictions = self.KNN_model.predict(test_point)
        # flatten list
        self.KNN_predictions = [item for sublist in self.KNN_predictions for item in sublist]
        # fetch and round the X and Y value
        posX = round(self.KNN_predictions[0], 2)
        posY = round(self.KNN_predictions[1], 2)
        # reassign values back into the array
        self.KNN_predictions[0], self.KNN_predictions[1] = posX, posY
        # return our predicted X,Y coordinate
        return self.KNN_predictions

    def model_evaluation(self, test_point):
        """
        Calculates the Margin of Error.
        :return: the Margin of Error.
        """
        # fetch the index number from X where our _test_point is located
        _ground_truth_index = np.where(np.all(X.values == test_point, axis=1))
        # fetch the ground_truth from the above index value
        _ground_truth_value = y.values[_ground_truth_index[0]]
        # model evaluation - returns margin of error in metres
        self.KNN_eval = mean_squared_error(_ground_truth_value.flatten(), self.KNN_predictions)
        return round(self.KNN_eval, 2)
