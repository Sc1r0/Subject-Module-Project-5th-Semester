# Data manipulation
import numpy as np  # used to calculate the square root in the euclidean distance function

# Files
from kNN_model_from_scratch import X, y


# calculate the euclidean distance
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    return np.sqrt(distance)
