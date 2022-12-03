# Data manipulation
import numpy as np

# Other
from euclidean_distance import euclidean_distance, euclidean_distance_multi


class KNNFromScratch:
    # FIXME: Our predictions are wrong.
    #   Our distances formula does not calculate the euclidean distance on itself at first, and thus probably returns
    #   wrong values all the way.
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, y):
        global predictions  # defined as global variable, to access it in estimated_location()

        predictions = self._predict(X, y)
        # print("Our predictions:", predictions)

        return np.array(predictions)

    def _predict(self, x, y):
        global k_indices  # defined as global variable, to access it in ground_truth()
        global groundtruth  # defined as global variable, to access it in ground_truth()

        y = np.array(y)
        distances = [euclidean_distance_multi(x, x_train) for x_train in self.X_train]
        # print("Our distances:", distances)
        # get closest K (indices = Latin for 'plural of index')
        k_indices = np.argsort(distances)[:self.k]  # returns index number of k_indices of the distances list
        # example: distance[66] = 39.42, which is the closest euclidean distance from the distances list.
        # print("Our k_indices:", k_indices)
        # print("Our k_indices:", distances[66], distances[47], distances[25], distances[67], distances[29])

        # save the ground truth locations
        groundtruth = y[k_indices]
        # print("Our groundtruths:", groundtruth)

        # save the k-nearest labels
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        return k_nearest_labels

    def ground_truth(self):
        ground_truth = groundtruth
        groundtruthx, groundtruthy = np.split(groundtruth, 2, axis=1)

        # flatten the 2-D list to a 1-D list
        groundtruthx = [item for sublist in groundtruthx for item in sublist]
        groundtruthy = [item for sublist in groundtruthy for item in sublist]

        # calculate the average of the X and Y values
        avg_x = sum(groundtruthx) / len(groundtruthx)
        avg_y = sum(groundtruthy) / len(groundtruthy)

        # create list to save the average ground truth for the X coordinate and Y coordinate
        avg_XY = [avg_x, avg_y]

        return ground_truth

    def margin_of_error(self, test, prediction):
        # TODO: Find euclidean distance from the ground_truth to the average of each index in prediction list
        # calculate the margin of error
        margin_of_error = euclidean_distance(test, prediction)

        # Don't mind this section:
        prediction = np.array(predictions)
        ground_truth = groundtruth

        print("prediction list:", prediction)
        print("ground_truth list:", ground_truth)

        average_euclidean_distance_singular = np.average(
            [np.sqrt((ground_truth[0][0] - prediction[0][0]) ** 2 +
                     (ground_truth[0][1] - prediction[0][1]) ** 2)])

        # FIXME: needs to be converted into a for-loop will make it dynamic instead of depending on the array
        #  length being 5.
        average_euclidean_distance = np.average(
            # first row
            [np.sqrt((ground_truth[0][0] - prediction[0][0]) ** 2 +
                     (ground_truth[0][1] - prediction[0][1]) ** 2),
             # second row
             np.sqrt((ground_truth[1][0] - prediction[1][0]) ** 2 +
                     (ground_truth[1][1] - prediction[1][1]) ** 2),
             # third row
             np.sqrt((ground_truth[2][0] - prediction[2][0]) ** 2 +
                     (ground_truth[2][1] - prediction[2][1]) ** 2),
             # fourth row
             np.sqrt((ground_truth[3][0] - prediction[3][0]) ** 2 +
                     (ground_truth[3][1] - prediction[3][1]) ** 2),
             # fifth row
             np.sqrt((ground_truth[4][0] - prediction[4][0]) ** 2 +
                     (ground_truth[4][1] - prediction[4][1]) ** 2)]
        )

        print("average euclidean distance between two points:", average_euclidean_distance_singular)
        print("average euclidean distance between multiple points:", average_euclidean_distance)
        # end of section

        return margin_of_error
