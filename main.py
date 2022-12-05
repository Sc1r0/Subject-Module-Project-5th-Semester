# Files
from KNN import KNNFromScratch as kNN
from MainWindow import window as window
from KNN_v2 import KNN_v2 as KNN_v2
# from excel_sheet_data import X, y, X_train, X_test, y_train, y_test

# Data manipulation
import pandas as pd  # is used to read data from Excel sheet
import numpy as np

# KNN tools
from sklearn.neighbors import KNeighborsRegressor  # is used to calculate the nearest neighbor for KNN regression
from sklearn.model_selection import train_test_split  # is used to split our dataset into training and testing sets.
from sklearn.metrics import mean_squared_error  # is used to calculate the quality of our model

from excel_sheet_data import X_train, y_train, y

test_point = [-21, -25, -31, -22]

if __name__ == '__main__':
    # run our window
    window()

    """
    KNN = KNN_v2(5)
    KNN.fit(X_train.values, y_train.values)
    prediction = KNN.predict(test_point, y)
    print("prediction:", prediction)

    # KNN.evaluate_knn()
    
    ############################### KNN WITHOUT LIBRARIES ###############################
    # the reason for 5, can be seen if "dia_viz.visualize_k_value()" is run
    k = 5

    # instantiate our KNN model and give it a k value
    KNN = kNN(k=k)

    # fit our KNN model
    # 'values' takes only the values of the Excel sheet and puts it into an array / list
    KNN.fit(X_train.values, y_train.values)

    # predict values
    our_predictions = KNN.predict(test_point, y)
    # print("predictions (our method): ", our_predictions)
    # print("Our y_test:", y_test.values[:k])

    # get our ground_truth variables
    ground_truth = KNN.ground_truth()

    # calculate the margin of error
    margin_of_error = KNN.margin_of_error(ground_truth, our_predictions)
    print("MoE (our method):", margin_of_error)
    #print("X,Y of distances[66] = ", y[65:68][:].values)

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
    """
