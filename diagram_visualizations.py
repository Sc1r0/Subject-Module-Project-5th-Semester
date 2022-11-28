# Data visualization
import matplotlib.pyplot as plt  # is used to plot the data from Excel sheet into a graph

# Files
import kNN_model as kNN


# prove the use of k = 5, using a scatter plot diagram
def visualize_k_value():
    margin_of_error = []
    for i in range(1, 15):
        knn_model_plotter = kNN.KNeighborsRegressor(i)
        knn_model_plotter.fit(kNN.X_train, kNN.y_train)
        predictions = knn_model_plotter.predict(kNN.X_test)
        margin_of_error.append(kNN.mean_squared_error(kNN.y_test, predictions))

    plt.title("Margin of error by k-value")
    plt.xlabel("k-value")
    plt.ylabel("margin of error (in meters)")
    plt.plot(range(1, 15), margin_of_error, marker='x', linestyle=':')
    for index, value in enumerate(margin_of_error):
        # offset the coordinates of the text above the 'x's and plot the value of 'x'
        plt.text(index + 0.7, value + 0.05, str(round(value, 2)))
    plt.show()
