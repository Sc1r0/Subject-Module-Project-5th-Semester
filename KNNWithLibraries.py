# KNN Libraries
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

# Our data
from excel_sheet_data import X_train, y_train, y_test


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

    def model_evaluation(self, prediction, groundtruth_counterpart):
        """
        Calculates the Margin of Error.
        :return: the Margin of Error.
        """
        # model evaluation - returns margin of error in meters
        self.KNN_eval = mean_squared_error(groundtruth_counterpart, prediction)

        return self.KNN_eval
