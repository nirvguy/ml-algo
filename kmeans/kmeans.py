import numpy as np
from . import init

def euclidean_distance(x, y):
    return np.linalg.norm(x-y)

class KMeans:
    """ K-means clustering """

    NOT_FITTED_ERROR_MESSAGE = 'Centroids not yet fitted to any data. Call to fit first.'

    def __init__(self, k, distance=euclidean_distance, init=init.forgy_initialization):
        """ Constructor

        Parameters
        ----------
          k : int
              Number of clusters to build
          distance : callable
              Distance (default: Euclidean distance)
          init : callable
              Centroids initialization method (default: kmeans.forgy_initialization)
        """
        self._k = k
        self._distance = distance
        self._centroids = []
        self._centroids_init = init

    def closest_cluster(self, x):
        """ Calculate the cluster closest to a point. ie
            the cluster with centroid that minimizes the distance to the point"""
        centroid_closest_idx = None
        centroid_min_dist = None

        for i, centroid in enumerate(self._centroids):
            dist = self._distance(x, centroid)
            if centroid_closest_idx is None or dist < centroid_min_dist:
                centroid_closest_idx = i
                centroid_min_dist = dist

        return centroid_closest_idx

    def _group_in_closest_clusters(self, data):
        """ Group every point of data in the closest cluster
 
        Returns
        -------
        bool
            True if any reassigment of clusters has changed. False otherwise.
        """
        any_changes = False

        for x_index, x in enumerate(data):
            cluster_idx = self.closest_cluster(x)

            prev_idx = self._cluster_indexes[x_index]
            self._cluster_indexes[x_index] = cluster_idx

            any_changes |= (prev_idx is None) or prev_idx != cluster_idx

        return any_changes

    def _means_per_cluster(self, data):
        """ Compute the mean intra cluster """
        centroids_shape = (self._k,) + data.shape[1:]
        means = np.zeros(centroids_shape)
        totals = np.zeros(centroids_shape)

        for x_index, cluster_idx in enumerate(self._cluster_indexes):
            means[cluster_idx] += data[x_index]
            totals[cluster_idx] += 1

        return means/totals

    def fit(self, data):
        """ Fit the centroids to the data

        Parameters
        ----------
        data: numpy.ndarray or list of numpy.ndarrays
            Dataset of points
        """
        self._cluster_indexes = [None] * len(data) # Cluster per data point
        # Initialize centroids based on init function
        self._centroids = self._centroids_init(data, self._k)

        while self._group_in_closest_clusters(data):
            self._centroids = self._means_per_cluster(data)

        del self._cluster_indexes

    def predict(self, X):
        """ Cluster index per data point

        Parameters
        ----------

        X: numpy.ndarray or list of numpy.ndarrays
            Dataset of points

        Returns
        -------
        list
            Cluster index between 0 and K-1 per data point
        """
        if len(self._centroids) == 0:
            raise Exception(self.NOT_FITTED_ERROR_MESSAGE)

        results = []
        for x in X:
            closest_idx = self.closest_cluster(x)
            results.append(closest_idx)
        return results

    @property
    def centroids(self):
        return np.array(self._centroids)
