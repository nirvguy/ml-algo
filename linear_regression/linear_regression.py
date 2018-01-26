from abc import abstractmethod
import numpy as np

class LinearRegressionTrainer(object):
    """ Interface for LinearRegression trainers """
    def __init__(self, model):
        """ Constructor

        Parameters
        ----------
        model : instance of LinearRegression
            Linear Regression model
        """
        self._model = model

    @abstractmethod
    def step(self, X, Y):
        """ Perform a learning step

        Parameters
        ----------
            X : numpy.ndarray
                input dataset of shape BATCH_SIZE x n_inputs
            Y : numpy.ndarray
                outputs dataset of shape BATCH_SIZE x n_outputs

        Returns
        -------
            Weights to update
        """

class MLSTrainer(LinearRegressionTrainer):
    """ Trainer of LinearRegression using Minimum Least Squares method """

    def step(self, X, Y):
        P = np.linalg.pinv(np.matmul(X.T, X))
        weights = np.matmul(np.matmul(P, X.T), Y)
        return weights

class LinearRegression:
    """ Linear Regression Model"""

    def __init__(self, n_inputs, n_outputs, trainer=MLSTrainer):
        """ Constructor

        Parameters
        ----------
        n_inputs : int
            Number of inputs of model
        n_outputs : int
            Number of outputs of model
        trainer : `LinearRegressionTrainer` subclass
            Trainer class. ie: MLSTrainer (default)
        """
        self._n_inputs = n_inputs
        self._trainer = trainer(self)
        # Encode bias parameter in one extra row in weights matrix
        self._weights = np.random.uniform(size=(n_inputs + 1, n_outputs))

    @property
    def weights(self):
        return self._weights

    def fit(self, X, Y, *extra_args, **extra_kwargs):
        """ Fit model to data

        Parameters
        ----------
            X : numpy.ndarray
                input dataset of shape BATCH_SIZE x n_inputs
            Y : numpy.ndarray
                outputs dataset of shape BATCH_SIZE x n_outputs
        """
        if len(X.shape) == 1 or len(Y.shape) == 1:
            raise Exception("X and Y dimensions must be BATCH_SIZE x inputs and BATCH_SIZE x outputs respectively")

        # Add a column of ones for the bias parameters
        X = np.append(X, np.ones((X.shape[0],1)), axis=1)

        # Update weights from trainer calculated weights
        self._weights = self._trainer.step(X, Y, *extra_args, **extra_kwargs)

    def predict(self, X):
        """ Predict outputs

        Parameters
        ----------
            X : numpy.ndarray
                Input points of shape SAMPLES x n_inputs
        """
        # Add a column of ones for the bias parameters
        X = np.append(X, np.ones((X.shape[0],1)), axis=1)
        return np.matmul(X, self._weights)
