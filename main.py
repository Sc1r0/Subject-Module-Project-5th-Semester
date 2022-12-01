# Files
from KNN import KNNFromScratch as kNN
import GUI as GUI

# Data manipulation
import pandas as pd  # is used to read data from Excel sheet
import numpy as np

# KNN tools
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

test_point = X_test.values[0]

if __name__ == '__main__':

    # run our window
    # GUI.main()

    ############################### KNN WITHOUT LIBRARIES ###############################
    k = 5   # the reason for 5, can be seen if "dia_viz.visualize_k_value()" is run
    # instantiate our KNN model and give it a k value
    KNN = kNN(k=k)
    # fit our KNN model
    # 'values' takes only the values of the Excel sheet and puts it into an array / list
    KNN.fit(X_train.values, y_train.values)
    # predict values
    our_predictions = KNN.predict(X_test.values, y)
    print("predictions (our method): ", our_predictions)
    print("Our y_test:", y_test.values[:k])
    margin_of_error = KNN.margin_of_error(y_test.values, our_predictions)
    #print("MoE (our method):", margin_of_error)
    #print("X,Y of distances[66] = ", y[65:68][:].values)

    ground_truth = KNN.ground_truth()
    #print("Our ground_truth return:", ground_truth)

    RMSE = KNN.root_mean_squared_error(our_predictions, y_test.values[:k], k)
    print("RMSE (Our Method):", RMSE)

    ################################# KNN WITH LIBRARIES #################################
    k = 5  # the reason for 5, can be seen if "dia_viz.visualize_k_value()" is run
    knn_model = KNeighborsRegressor(n_neighbors=k)
    # instance and fit the model
    knn_model.fit(X_train, y_train)
    # predictions - returns predicted values
    knn_predictions = knn_model.predict(X_test)
    #print("predictions (KNN Library):", knn_predictions[:k])
    # model evaluation - returns margin of error in meters
    knn_eval = mean_squared_error(y_test, knn_predictions)
    print("RMSE (KNN Library):", knn_eval)

