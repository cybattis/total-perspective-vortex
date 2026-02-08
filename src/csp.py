import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class MyCSP(BaseEstimator, TransformerMixin):
    """
    Custom implementation of the Common Spatial Pattern (CSP) algorithm.

    The goal is to find spatial filters that maximize the variance
    for one class and minimize it for the other.
    """
    def __init__(self, n_components=4):
        """
        Args:
            n_components (int): Number of filters to keep (must be even).
                                We keep n/2 filters for each extreme class.
        """
        self.n_components = n_components
        self.filters = None

    @staticmethod
    def _compute_covariance_matrix(X):
        """
        Computes the average covariance matrix for a set of epochs.
        X shape: (n_epochs, n_channels, n_times)
        """
        # For each epoch, we calculate the spatial covariance: (C x T) @ (T x C) -> (C x C)
        # Normalize by the trace to avoid absolute amplitude dominance
        covs = []
        for epoch in X:
            c = np.dot(epoch, epoch.T)
            c /= np.trace(c) # Trace normalization (optional but recommended in CSP)
            covs.append(c)

        return np.mean(covs, axis=0)

    def fit(self, X, y):
        """
        Calculates the CSP spatial filters.
    
        Args:
            X: EEG data (n_epochs, n_channels, n_times)
            y: Labels (n_epochs,)
        """
        # Input validation (X must be 3D for EEG)
        if X.ndim != 3:
            raise ValueError("X must have 3 dimensions (epochs, channels, times)")
    
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("This CSP implementation only supports 2 classes.")
        
        # 1. Compute average covariances for each class
        covs = []
        for cls in classes:
            x_cls = X[y == cls]
            covs.append(self._compute_covariance_matrix(x_cls))
        
        R1 = covs[0]
        R2 = covs[1]
    
        # Regularization: Add a small epsilon to the diagonal to ensure positive definiteness
        cov_sum = R1 + R2
        cov_sum += np.eye(cov_sum.shape[0]) * 1e-8

        # 2. Generalized eigenvalue problem resolution
        # We look for w such that: R1 * w = lambda * (R1 + R2) * w
        # This amounts to maximizing the variance of R1 relative to the total variance
        # eigh is used for hermitian/symmetric matrices
        eigen_values, eigen_vectors = eigh(R1, cov_sum)

        # 3. Sort eigenvectors (descending order of eigenvalues)
        # eigh returns ascending order, so we reverse it
        ix = np.argsort(np.abs(eigen_values))[::-1]
        sorted_vectors = eigen_vectors[:, ix]
    
        # 4. Filter selection
        # We take the first n/2 (maximize class 1) and last n/2 (maximize class 2)
        n_pick = self.n_components // 2
    
        filters_top = sorted_vectors[:, :n_pick]
        filters_bottom = sorted_vectors[:, -n_pick:]

        # Concatenate and transpose to get a projection matrix (n_comp, n_chans)
        self.filters = np.concatenate([filters_top, filters_bottom], axis=1).T

        return self

    def transform(self, X):
        """
        Applies spatial filters and computes log-variance.
    
        Args:
            X: EEG data (n_epochs, n_channels, n_times)
        Returns:
            X_transformed: Features (n_epochs, n_components)
        """
        check_is_fitted(self, ["filters"])

        # X shape: (N, C, T), Filters shape: (K, C)
        # We want to project: Z = W * X -> (K, T) for each epoch

        # np.tensordot or explicit loop. Using matmul for clarity.
        # X is (N, C, T), filters_.T is (C, K). We want (N, K, T)
        # We can do: (filters_ @ X_epoch)

        x_filtered = np.asarray([np.dot(self.filters, epoch) for epoch in X])

        # Compute power (variance) of the filtered signal
        # Var(Z) = mean(Z^2) because signals are generally centered (band-pass)
        x_power = np.mean(x_filtered**2, axis=2)
    
        # Log-transformation to normalize distribution (Log-Normal -> Normal)
        x_features = np.log(x_power)
    
        return x_features
