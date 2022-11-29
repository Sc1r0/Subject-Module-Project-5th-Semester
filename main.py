# Files
from KNN import KNNFromScratch as kNN

# Data manipulation
import pandas as pd  # is used to read data from Excel sheet

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

if __name__ == '__main__':
    ############################### KNN WITHOUT LIBRARIES ###############################
    # instantiate our KNN model and give it a k value
    reg = kNN(k=5)
    # fit our KNN model
    # 'values' takes only the values of the Excel sheet and puts it into an array / list
    reg_fit = reg.fit(X_train.values, y_train.values)
    # predict values
    our_predictions = reg.predict(X_test.values)
    print("predictions (our method): ", our_predictions)
    margin_of_error = reg.margin_of_error(y_test.values[0], our_predictions[0, 0])
    print("MoE (our method):", margin_of_error)

    ################################# KNN WITH LIBRARIES #################################
    k = 5  # the reason for 5, can be seen if "dia_viz.visualize_k_value()" is run
    knn_model = KNeighborsRegressor(n_neighbors=k)
    # instance and fit the model
    knn_model.fit(X_train, y_train)
    # predictions - returns predicted values
    knn_predictions = knn_model.predict(X_test)
    print("predictions (KNN Library):", knn_predictions[:k])
    # model evaluation - returns margin of error in meters
    knn_eval = mean_squared_error(y_test, knn_predictions)
    print("MoE (KNN Library):", knn_eval)

