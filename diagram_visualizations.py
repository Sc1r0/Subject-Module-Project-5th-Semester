# Data visualization
import matplotlib.pyplot as plt  # is used to plot the data from Excel sheet into a graph
import numpy as np

# Files
from KNNFromScratch import KNNFromScratch
from excel_sheet_data import X_train, y_train, X_test, X, y


# prove the use of k = 5, using a scatter plot diagram
def visualize_k_value():
    """
    A method to create a graph, plotting the error margin of the selected k, from 1 to 15. It calculates the best
    k-value by using the predict() method with X_test.values as the test_points and y_train as the ground_truths.
    :return: A graph showing the margin of error for k-value 1 to 15.
    """
    plt.figure(figsize=(16, 10))

    margin_of_error = []
    for i in range(1, 15):
        KNN = KNNFromScratch(k=i)
        KNN.fit(X_train.values, y_train.values)
        our_predictions = KNN.predict(X_test.values, y_train.values)
        margin_of_error.append(KNN.evaluate_knn(our_predictions))

    plt.title("Margin of error by k-value")
    plt.xlabel("k-value")
    plt.ylabel("margin of error (in meters)")
    plt.plot(range(2, 15), margin_of_error, marker='x', linestyle=':')
    for index, value in enumerate(margin_of_error):
        # offset the coordinates of the text above the 'x's and plot the value of 'x'
        plt.text(index + 1.85, value + 0.015, str(round(value, 2)))

    # plt.show()


def euclidean_distances(distances, k_value):
    plt.figure(figsize=(16, 10))

    # all euclidean distances
    plt.subplot(1, 2, 1)
    plt.plot(distances, marker='o')
    plt.title("Euclidean distances from Test_Point to each row in X_train")
    plt.xlabel("Index value in X_train")
    plt.ylabel("distance from test_point")
    plt.grid()

    # k-nearest euclidean distances
    distances_sorted = np.argsort(distances)[:k_value]
    plt.subplot(1, 2, 2)
    plt.plot(distances_sorted, marker='o')
    plt.title("K-nearest Index in X_Train")
    plt.xlabel("k-value")
    plt.ylabel("Index Number")
    plt.grid()

    # plt.show()


def dataset_RSSI():
    # beacon values
    beacons = np.array(X)
    b1 = beacons[:, 0]
    b2 = beacons[:, 1]
    b3 = beacons[:, 2]
    b4 = beacons[:, 3]
    fontsize = 12

    plt.figure(figsize=(16, 10))

    # beacon 2
    plt.subplot(2, 2, 1)
    plt.plot(b1, marker='o', color='red')
    plt.xlim(0, 100)
    plt.ylim(-10, -75)
    plt.title("Beacon 2 RSSI values")
    plt.xlabel("Index Value", fontsize=fontsize)
    plt.ylabel("RSSI value", fontsize=fontsize)
    plt.grid()

    # beacon 5
    plt.subplot(2, 2, 2)
    plt.plot(b4, marker='o', color='green')
    plt.xlim(0, 100)
    plt.ylim(-10, -75)
    plt.title("Beacon 5 RSSI values")
    plt.xlabel("Index Value", fontsize=fontsize)
    plt.ylabel("RSSI value", fontsize=fontsize)
    plt.grid()

    # beacon 3
    plt.subplot(2, 2, 3)
    plt.plot(b2, marker='o', color='blue')
    plt.xlim(0, 100)
    plt.ylim(-10, -75)
    plt.title("Beacon 3 RSSI values")
    plt.xlabel("Index Value", fontsize=fontsize)
    plt.ylabel("RSSI value", fontsize=fontsize)
    plt.grid()

    # beacon 4
    plt.subplot(2, 2, 4)
    plt.plot(b3, marker='o', color='orange')
    plt.xlim(0, 100)
    plt.ylim(-10, -75)
    plt.title("Beacon 4 RSSI values")
    plt.xlabel("Index Value", fontsize=fontsize)
    plt.ylabel("RSSI value", fontsize=fontsize)
    plt.grid()

    # plt.show()

# dataset_RSSI()
# euclidean_distances()
# visualize_k_value()
