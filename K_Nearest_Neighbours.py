import numpy as np


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

        - Take origin point

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

        print(self.k_min_distances)
        print(self.k_classes)


# Launch KNN
if __name__ == "__main__":
    # Create class instance
    k = int(input("Write k: "))
    knn = KNN(k)

    # Generate random dataset
    origin = np.random.rand(1, 5)
    x_train = np.random.randint(100, size=(10, 5))
    y_train = np.random.randint(3, size=(10, 1))

    # Get normalisation
    normalisation_number_x = np.linalg.norm(x_train)

    # Normalise datasets
    x_train = x_train / normalisation_number_x

    knn.fit(x_train, y_train)
    knn.predict(origin[0])

    # print(x_train)
