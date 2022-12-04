# Data visualization
import matplotlib.pyplot as plt  # is used to plot the data from Excel sheet into a graph

# Files
from KNN import KNNFromScratch as kNN
from excel_sheet_data import X_train, y_train, X_test, y_test

# prove the use of k = 5, using a scatter plot diagram
def visualize_k_value():
    margin_of_error = []
    for i in range(1, 15):
        KNN = kNN(k=i)
        KNN.fit(X_train.values, y_train.values)
        our_predictions = KNN.predict(X_test.values)
        margin_of_error.append(KNN.margin_of_error(y_test.values, our_predictions))

        # knn_model_plotter = KNeighborsRegressor(i)
        # knn_model_plotter.fit(X_train, y_train)
        # predictions = knn_model_plotter.predict(X_test)
        # margin_of_error.append(mean_squared_error(y_test, predictions))

    plt.title("Margin of error by k-value")
    plt.xlabel("k-value")
    plt.ylabel("margin of error (in meters)")
    plt.plot(range(1, 15), margin_of_error, marker='x', linestyle=':')
    for index, value in enumerate(margin_of_error):
        # offset the coordinates of the text above the 'x's and plot the value of 'x'
        plt.text(index + 0.7, value + 0.05, str(round(value, 2)))
    plt.show()


visualize_k_value()
