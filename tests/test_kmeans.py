import unittest
import numpy as np
import numpy.testing as np_test
from kmeans import KMeans
from kmeans.kmeans import euclidean_distance
from kmeans import init
import math

class KmeansTest(unittest.TestCase):
    def test01_non_fitted_returns_no_centroids(self):
        model = KMeans(k=2)
        self.assertEqual(len(model.centroids), 0)

    def test02_non_fitted_model_raises_not_fitted_error_message(self):
        model = KMeans(k=2)
        try:
            model.predict(np.array([[1,0], [0, 1]]))
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), KMeans.NOT_FITTED_ERROR_MESSAGE)

    def test03_distances(self):
        self.assertEqual(euclidean_distance(np.array([0, 0]), np.array([0.0])), 0.0)
        self.assertAlmostEqual(euclidean_distance(np.array([1, 0]), np.array([2, 0])), 1.0)
        self.assertAlmostEqual(euclidean_distance(np.array([1, 1]), np.array([0, 0])), math.sqrt(2))

    def test04_forgy_initialization(self):
        np.random.seed(5)
        data = np.array([[1, 0], [0, 0]])
        centroids = init.forgy_initialization(data, 2)
        np_test.assert_array_equal(centroids, np.array([[0, 0], [1, 0]]))

    def test05_fit_one_cluster(self):
        model = KMeans(k=1, init=init.forgy_initialization)

        data = np.array([[0.0, 0.0]])

        model.fit(data)

        self.assertEqual(model.predict(data), [0])

        np_test.assert_array_equal(model.centroids, np.array([[0.0, 0.0]]))

        test_points = np.array([[1.0, 0.0],
                                [0.0, 1.0],
                                [1.0, 1.0]])
        self.assertEqual(model.predict(test_points), [0] * 3)

    def test06_fit_two_clusters(self):
        np.random.seed(1)
        model = KMeans(k=2, init=init.forgy_initialization)
        data = np.array([[-1.0, 0.0], [-1.001, 0.0], [-0.999, 0.0],
                         [0.0, 1.0], [0.0, 0.999], [0.0, 1.001]])

        model.fit(data)
        self.assertEquals(model.predict(data), [1,1,1,0,0,0])
