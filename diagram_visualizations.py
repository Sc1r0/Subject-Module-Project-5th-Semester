# Data visualization
import matplotlib.pyplot as plt  # is used to plot the data from Excel sheet into a graph

# Files
from KNNFromScratch import KNNFromScratch
from excel_sheet_data import X_train, y_train, X_test


# prove the use of k = 5, using a scatter plot diagram
class best_K_value:
    def visualize_k_value():
        """
        A method to create a graph, plotting the error margin of the selected k, from 1 to 15. It calculates the best
        k-value by using the predict() method with X_test.values as the test_points and y_train as the ground_truths.
        :return: A graph showing the margin of error for k-value 1 to 15.
        """
        margin_of_error = []
        for i in range(1, 15):
            KNN = KNNFromScratch(k=i)
            KNN.fit(X_train.values, y_train.values)
            our_predictions = KNN.predict(X_test.values, y_train)
            margin_of_error.append(KNN.evaluate_knn(our_predictions))

        """
        plt.title("Margin of error by k-value")
        plt.xlabel("k-value")
        plt.ylabel("margin of error (in meters)")
        plt.plot(range(1, 15), margin_of_error, marker='x', linestyle=':')
        for index, value in enumerate(margin_of_error):
            # offset the coordinates of the text above the 'x's and plot the value of 'x'
            plt.text(index + 0.7, value + 0.05, str(round(value, 2)))
        plt.show()
        """

        return margin_of_error
