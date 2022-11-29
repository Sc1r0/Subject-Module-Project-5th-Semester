# Data manipulation
import numpy as np  # used to calculate the square root in the euclidean distance function


# method to calculate the distance between two rows in the dataset
def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance
