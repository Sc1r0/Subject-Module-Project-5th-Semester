# Data manipulation
import numpy as np  # used to calculate the square root in the euclidean distance function


# method to calculate the margin of error in KNN.py
def euclidean_distance(row1, row2):
    distance = np.sqrt(np.sum(row1[0, 0] - row2[0, 0]) ** 2)
    return distance


# method to calculate the euclidean distance between multiple columns
def euclidean_distance_multi(row1, row2):
    distance = np.sqrt(np.sum(
        ((row1[0] - row2[0]) ** 2) +
        ((row1[1] - row2[1]) ** 2) +
        ((row1[2] - row2[2]) ** 2) +
        ((row1[3] - row2[3]) ** 2)
        )
    )
    return distance
