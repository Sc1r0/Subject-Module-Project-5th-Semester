# Data manipulation
import pandas as pd  # is used to read data from Excel sheet

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

# instance and fit the model, n_neighbors = 5 has been chosen, as it gave the best result (0.6198).
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# calculate the score
score_knn = knn_model.score(X_test, y_test)

# predictions - also returns the accuracy of our model; 1.2 = 1.2 meters
prediction = knn_model.predict(X_test)

# model evaluation
knn_eval = mean_squared_error(y_test, prediction)