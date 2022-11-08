# Files
import kNN_model as kNN
import diagram_visualizations as dia_viz
from euclidean_distance import euclidean_distance


if __name__ == '__main__':

    distances = []
    for i in range(len(kNN.X)):
        ed_values = euclidean_distance(kNN.X.iloc[i], i)
        distances.append(ed_values)

    print(distances)

    dia_viz.visualize_k_value()
    """
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
