"""
Temporal Regularized Matrix Factorization
"""

# Author: Alexander Semenov <alexander.s.semenov@yandex.ru>

import numpy as np

# TRMF model
lambda_f = 1.
lambda_x = 1.
lambda_w = 1.
eta = 1.
alpha = 1000.
max_iter = 5000
K = 8
lags = [1, 24]


class trmf:
    """Temporal Regularized Matrix Factorization.

    Parameters
    ----------
    
    lags : array-like, shape (n_lags,)
        Set of lag indices to use in model.
    
    K : int
        Length of latent embedding dimension
    
    lambda_f : float
        Regularization parameter used for matrix F.
    
    lambda_x : float
        Regularization parameter used for matrix X.
    
    lambda_w : float
        Regularization parameter used for matrix W.

    alpha : float
        Regularization parameter used for make the sum of lag coefficient close to 1.
        That helps to avoid big deviations when forecasting.
    
    eta : float
        Regularization parameter used for X when undercovering autoregressive dependencies.

    max_iter : int
        Number of iterations of updating matrices F, X and W.

    F_step : float
        Step of gradient descent when updating matrix F.

    X_step : float
        Step of gradient descent when updating matrix X.

    W_step : float
        Step of gradient descent when updating matrix W.


    Attributes
    ----------

    F : ndarray, shape (n_timeseries, K)
        Latent embedding of timeseries.

    X : ndarray, shape (K, n_timepoints)
        Latent embedding of timepoints.

    W : ndarray, shape (K, n_lags)
        Matrix of autoregressive coefficients.
    """

    def __init__(self, lags, K, lambda_f, lambda_x, lambda_w, alpha, eta, max_iter=1000,
                 F_step=0.0001, X_step=0.0001, W_step=0.0001):
        self.lags = lags
        self.L = len(lags)
        self.K = K
        self.lambda_f = lambda_f
        self.lambda_x = lambda_x
        self.lambda_w = lambda_w
        self.alpha = alpha
        self.eta = eta
        self.max_iter = max_iter
        self.F_step = F_step
        self.X_step = X_step
        self.W_step = W_step

        self.W = None
        self.F = None
        self.X = None

    def fit(self, train, resume=False):
        """Fit the TRMF model according to the given training data.

        Model fits through sequential updating three matrices:
            -   matrix self.F;
            -   matrix self.X;
            -   matrix self.W.
            
        Each matrix updated with gradient descent.

        Parameters
        ----------
        train : ndarray, shape (n_timeseries, n_timepoints)
            Training data.

        resume : bool
            Used to continue fitting.

        Returns
        -------
        self : object
            Returns self.
        """

        if not resume:
            self.Y = train
            mask = np.array((~np.isnan(self.Y)).astype(int))
            self.mask = mask
            self.Y[self.mask == 0] = 0.
            self.N, self.T = self.Y.shape
            self.W = np.random.randn(self.K, self.L) / self.L
            self.F = np.random.randn(self.N, self.K)
            self.X = np.random.randn(self.K, self.T)

        for _ in range(self.max_iter):
            self._update_F(step=self.F_step)
            self._update_X(step=self.X_step)
            self._update_W(step=self.W_step)

    def predict(self, h):
        """Predict each of timeseries h timepoints ahead.

        Model evaluates matrix X with the help of matrix W,
        then it evaluates prediction by multiplying it by F.

        Parameters
        ----------
        h : int
            Number of timepoints to forecast.

        Returns
        -------
        preds : ndarray, shape (n_timeseries, T)
            Predictions.
        """

        X_preds = self._predict_X(h)
        return np.dot(self.F, X_preds)

    def _predict_X(self, h):
        """Predict X h timepoints ahead.

        Evaluates matrix X with the help of matrix W.

        Parameters
        ----------
        h : int
            Number of timepoints to forecast.

        Returns
        -------
        X_preds : ndarray, shape (self.K, h)
            Predictions of timepoints latent embeddings.
        """

        X_preds = np.zeros((self.K, h))
        X_adjusted = np.hstack([self.X, X_preds])
        for t in range(self.T, self.T + h):
            for l in range(self.L):
                lag = self.lags[l]
                X_adjusted[:, t] += X_adjusted[:, t - lag] * self.W[:, l]
        return X_adjusted[:, self.T:]

    def impute_missings(self):
        """Impute each missing element in timeseries.

        Model uses matrix X and F to get all missing elements.

        Parameters
        ----------

        Returns
        -------
        data : ndarray, shape (n_timeseries, T)
            Predictions.
        """
        data = self.Y
        data[self.mask == 0] = np.dot(self.F, self.X)[self.mask == 0]
        return data

    def _update_F(self, step, n_iter=1):
        """Gradient descent of matrix F.

        n_iter steps of gradient descent of matrix F.

        Parameters
        ----------
        step : float
            Step of gradient descent when updating matrix.

        n_iter : int
            Number of gradient steps to be made.

        Returns
        -------
        self : objects
            Returns self.
        """

        for _ in range(n_iter):
            self.F -= step * self._grad_F()

    def _update_X(self, step, n_iter=1):
        """Gradient descent of matrix X.

        n_iter steps of gradient descent of matrix X.

        Parameters
        ----------
        step : float
            Step of gradient descent when updating matrix.

        n_iter : int
            Number of gradient steps to be made.

        Returns
        -------
        self : objects
            Returns self.
        """

        for _ in range(n_iter):
            self.X -= step * self._grad_X()

    def _update_W(self, step, n_iter=1):
        """Gradient descent of matrix W.

        n_iter steps of gradient descent of matrix W.

        Parameters
        ----------
        step : float
            Step of gradient descent when updating matrix.

        n_iter : int
            Number of gradient steps to be made.

        Returns
        -------
        self : objects
            Returns self.
        """

        for _ in range(n_iter):
            self.W -= step * self._grad_W()

    def _grad_F(self):
        """Gradient of matrix F.

        Evaluating gradient of matrix F.

        Parameters
        ----------

        Returns
        -------
        self : objects
            Returns self.
        """

        return - 2 * np.dot((self.Y - np.dot(self.F, self.X)) * self.mask,
                            self.X.T) + 2 * self.lambda_f * self.F

    def _grad_X(self):
        """Gradient of matrix X.

        Evaluating gradient of matrix X.

        Parameters
        ----------

        Returns
        -------
        self : objects
            Returns self.
        """

        for l in range(self.L):
            lag = self.lags[l]
            W_l = self.W[:, l].repeat(self.T, axis=0).reshape(self.K, self.T)
            X_l = self.X * W_l
            z_1 = self.X - np.roll(X_l, lag, axis=1)
            z_1[:, :max(self.lags)] = 0.
            z_2 = - (np.roll(self.X, -lag, axis=1) - X_l) * W_l
            z_2[:, -lag:] = 0.

        grad_T_x = z_1 + z_2
        return - 2 * np.dot(self.F.T, self.mask * (
                    self.Y - np.dot(self.F, self.X))) + self.lambda_x * grad_T_x + self.eta * self.X

    def _grad_W(self):
        """Gradient of matrix W.

        Evaluating gradient of matrix W.

        Parameters
        ----------

        Returns
        -------
        self : objects
            Returns self.
        """

        grad = np.zeros((self.K, self.L))
        for l in range(self.L):
            lag = self.lags[l]
            W_l = self.W[:, l].repeat(self.T, axis=0).reshape(self.K, self.T)
            X_l = self.X * W_l
            z_1 = self.X - np.roll(X_l, lag, axis=1)
            z_1[:, :max(self.lags)] = 0.
            z_2 = - (z_1 * np.roll(self.X, lag, axis=1)).sum(axis=1)
            grad[:, l] = z_2
        return grad + self.W * 2 * self.lambda_w / self.lambda_x - \
               self.alpha * 2 * (1 - self.W.sum(axis=1)).repeat(self.L).reshape(self.W.shape)
