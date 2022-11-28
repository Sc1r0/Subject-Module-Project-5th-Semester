# Data manipulation
import numpy as np
import pandas as pd  # is used to read data from Excel sheet

# out-commented to avoid having unnecessary functions or processes done
"""
# Modeling
from sklearn.neighbors import KNeighborsRegressor  # is used to calculate the nearest neighbor for KNN regression
from sklearn.model_selection import train_test_split  # is used to split our dataset into training and testing sets.
from sklearn.metrics import mean_squared_error  # is used to calculate the quality of our model

# fetch the Excel file
dataset = pd.read_excel("Marvelmind(X,Y,RSSI).xlsx", sheet_name="Square")

# X => dependent values, y => independent values
X = dataset[['b2', 'b3', 'b4', 'b5']]  # the RSSI values from the beacons
y = dataset[['X', 'Y']]  # our X and Y coordinates from the modem

# train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
"""


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


class KNN:
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


# out-commented to avoid having unnecessary functions or processes done
"""
# instance and fit the model, n_neighbors = 5 has been chosen, as it gave the best result (0.6198).
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# calculate the score
score_knn = knn_model.score(X_test, y_test)

# predictions - also returns the accuracy of our model; 1.2 = 1.2 meters
prediction = knn_model.predict(X_test)

# model evaluation
knn_eval = mean_squared_error(y_test, prediction)
"""
