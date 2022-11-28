# Files
from kNN_model_from_scratch import KNNFromScratch
import diagram_visualizations as dia_viz

# Data manipulation
import numpy as np
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


################################# KNN WITH LIBRARIES #################################
k = 5   # the reason for 5, can be seen if "dia_viz.visualize_k_value()" is run
knn_model = KNeighborsRegressor(n_neighbors=k)
# instance and fit the model
knn_model.fit(X_train, y_train)
# predictions - returns predicted values
prediction = knn_model.predict(X_test)
# model evaluation - returns margin of error in meters
knn_eval = mean_squared_error(y_test, prediction)

############################### KNN WITHOUT LIBRARIES ###############################
# instantiate our KNN model and give it a k value
reg = KNNFromScratch(k=5)
# fit our KNN model
reg.fit(X_train, y_train)
# predict values
predictions = reg.predict(X_test)


if __name__ == '__main__':
    print("Predictions:")
    print(prediction)   # print predictions
    print("Mean squared error:", knn_eval)  # print the evaluation

    # dia_viz.visualize_k_value()

    # out-commented to avoid having unnecessary functions or processes done
    """
    distances = []
    for i in range(len(kNN.X)):
        ed_values = kNN.euclidean_distance_v2(kNN.X.iloc[i], i)
        distances.append(ed_values)

    print(distances)

    print(f'Number of rows: {kNN.dataset.shape[0]} | Columns (variables): {kNN.dataset.shape[1]}\n')
    print(f'Train: {kNN.X_train.shape, kNN.y_train.shape} \nTest: {kNN.X_test.shape, kNN.y_test.shape}')
    print("Margin of error (in meters):", kNN.knn_eval, "\n")

    print("predictions (X, Y):\n")
    print(kNN.prediction, "\n")

    print("y_test (X, Y):\n")
    print(kNN.y_test, "\n")

    if kNN.score_knn < 0.95:
        print("The score:", kNN.score_knn)
        print("That's not very precise. We'll be flying all over the place!")

    else:
        print("The score:", kNN.score_knn)
        print("That's really good! We'll have a very solid approximation of our location!")
    """
