# Data visualization
import matplotlib.pyplot as plt  # is used to plot the data from Excel sheet into a graph

# Files
import kNN_model as kNN


# prove the use of k = 5, using a scatter plot diagram
def visualize_k_value():
    accuracy_vals = []
    for i in range(1, 15):
        knn_model_plotter = kNN.KNeighborsRegressor(i)
        knn_model_plotter.fit(kNN.X_train, kNN.y_train)
        predictions = knn_model_plotter.predict(kNN.X_test)
        accuracy_vals.append(kNN.mean_squared_error(kNN.y_test, predictions))

    plt.title("Average margin of error according to the k-value")
    plt.xlabel("k-value")
    plt.ylabel("margin of error (in meters)")
    plt.plot(range(1, 15), accuracy_vals, marker='x', linestyle=':')
    plt.show()


