import numpy as np
from matplotlib import pyplot as plt


def sort_dict(dictionary):
    """
    Simple bubble sort, but with addition of sorting the keys depending on their values
    :param dictionary: dictionary to sort
    :return: dictionary
    """

    dict_keys = list(dictionary.keys())
    dict_values = list(dictionary.values())

    for i in range(len(dict_values) - 1):
        for j in range(i + 1, len(dict_values)):
            if dict_values[i] > dict_values[j]:
                copy_values = [dict_keys[i], dict_values[i]]

                dict_keys[i] = dict_keys[j]
                dict_values[i] = dict_values[j]

                dict_keys[j] = copy_values[0]
                dict_values[j] = copy_values[1]

    return dict(zip(dict_keys, dict_values))


class KNN:
    def __init__(self, k):
        super().__init__()

        self.k = k
        self.k_min_distances = []
        self.k_classes = []

        self.all_distances = []

        self.x_train = np.array([])
        self.y_train = np.array([])

        # Number of rows of x_train
        self.x_dimensionality = 0

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

        self.x_dimensionality = np.shape(x)[0]

    def predict(self, origin):
        """
        :param origin - the point which class we want to predict
        :return: none
        """

        # Find cosine similarity distance metric between each individual and given origin point
        for i in range(self.x_dimensionality):
            point_in_space = self.x_train[i]
            cosine_similarity = np.dot(point_in_space, origin) / (np.linalg.norm(point_in_space) *
                                                                  np.linalg.norm(origin))

            self.all_distances.append(cosine_similarity)

        # Find k points with minimal distances
        all_dist_copy = self.all_distances.copy()
        for j in range(self.k):
            min_point = min(all_dist_copy)
            min_index = self.all_distances.index(min_point)

            self.k_min_distances.append(min_point)
            self.k_classes.append(self.y_train[min_index][0])

            all_dist_copy.remove(min_point)

        classes_count = self.count_classes_in_list(self.k_classes)
        sorted_classes_count = sort_dict(classes_count)

        # print("Min distances: ", self.k_min_distances)
        # print("Min distances' classes ", self.k_classes)
        print("Your point has class: ", list(sorted_classes_count.keys())[-1])

    def count_classes_in_list(self, classes_list):
        """
        :param classes_list: classes of points which have minimal distance to origin
        :return: classes_count: sorted list of classes by their quantity in classes_list
        """

        classes_count = dict()
        for i in range(len(classes_list)):
            key = str(classes_list[i])
            if key in classes_count.keys():
                classes_count[key] += 1
            else:
                classes_count[key] = 1

        return classes_count


def plot_color_points(x_train, y_train):
    """
    Plots a graph with given points from x_train and with colors (classes) from y_train
    :param x_train: 2D points
    :param y_train: points classes
    :return: none
    """
    plt.scatter(x_train[:, :1], x_train[:, 1:2])
    plt.show()
    plt.show()


# Launch KNN
if __name__ == "__main__":
    # Create class instance
    k = int(input("Write k: "))
    knn = KNN(k)

    # Generate random dataset
    origin = np.random.rand(1, 2)
    x_train = np.random.randint(100, size=(100, 2))
    y_train = np.random.randint(10, size=(100, 1))

    # Get normalisation
    normalisation_number_x = np.linalg.norm(x_train)

    # Normalise datasets
    x_train = x_train / normalisation_number_x

    knn.fit(x_train, y_train)
    knn.predict(origin[0])

    plot_color_points(x_train, y_train)
