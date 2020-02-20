import numpy as np


class LinearRegressor(object):
    """Linear Regressor object.

    Parameters
    ----------
    X: np.ndarray.
        Co-variates vector.
    Y: np.ndarray.
        Vector of responses.
    """
    _eps = 1e-12  # numerical precision

    def __init__(self, X, Y):
        self._Xtr = X
        self._Ytr = Y
        self._w = np.random.randn(X.shape[1])

    @property
    def number_samples(self):
        """Return length of training set."""
        return self._Xtr.shape[0]

    @property
    def weights(self):
        """Get weights of regressor object."""
        return self._w

    @weights.setter
    def weights(self, value):
        self._w = value

    def load_data(self, X, Y=None):
        """Load training data."""
        self._Xtr = X
        self._Ytr = Y

    def close_form_weights(self):
        """Caluclate the weights using the closed form expression."""
        dim = self._Xtr.shape[1]
        self._w = np.dot(
            np.linalg.pinv(np.dot(self._Xtr.T, self._Xtr) + self._eps * np.eye(dim)),
            np.dot(self._Xtr.T, self._Ytr))
        return self._w

    def predict(self, X):
        """Predict an output given the inptus."""
        return np.dot(X, self.weights)

    def test_loss(self, w, X, Y):
        """Calculate the test loss with a different w."""
        w_old = self.weights
        self.weights = w
        error = self.predict(X) - Y

        self.weights = w_old
        return np.dot(error.T, error)

    def loss(self, w, indexes=None):
        """Get loss of w and the current index."""
        if indexes is None:
            indexes = np.arange(self.number_samples)

        self.weights = w
        error = self.predict(self._Xtr[indexes, :]) - self._Ytr[indexes]
        return np.dot(error.T, error) / indexes.size

    def gradient(self, w, indexes=None):
        """Get gradient of w and the current index."""
        if indexes is None:
            indexes = np.arange(self.number_samples)

        self.weights = w
        error = self.predict(self._Xtr[indexes, :]) - self._Ytr[indexes]
        return np.dot(self._Xtr[indexes, :].T, error) / indexes.size
