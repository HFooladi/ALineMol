from sklearn.ensemble import RandomForestClassifier


class CML:
    """Classical machine-learning baseline classifier.

    A thin wrapper around :class:`sklearn.ensemble.RandomForestClassifier` that
    exposes the standard scikit-learn estimator API (``fit``/``predict``/
    ``predict_proba``/``score``/``get_params``/``set_params``). It provides a
    simple, dependency-light baseline for molecular property classification that
    can be compared against the graph neural network models in
    :mod:`alinemol.models.fragGNN`.

    Attributes:
        model: The underlying ``RandomForestClassifier`` instance
            (100 trees, ``random_state=0``).

    Example:
        >>> from alinemol.models.cml import CML
        >>> clf = CML()
        >>> clf.fit(X_train, y_train)  # doctest: +SKIP
        >>> preds = clf.predict(X_test)  # doctest: +SKIP
    """

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=0)

    def fit(self, X, y):
        """Fit the random forest on training features and labels.

        Args:
            X (numpy.ndarray): Feature matrix of shape ``(n_samples, n_features)``.
            y (numpy.ndarray): Target labels of shape ``(n_samples,)``.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """Predict class labels for ``X``.

        Args:
            X (numpy.ndarray): Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            numpy.ndarray: Predicted class labels of shape ``(n_samples,)``.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities for ``X``.

        Args:
            X (numpy.ndarray): Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            numpy.ndarray: Array of shape ``(n_samples, n_classes)`` with
            per-class probabilities.
        """
        return self.model.predict_proba(X)

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Args:
            X (numpy.ndarray): Feature matrix of shape ``(n_samples, n_features)``.
            y (numpy.ndarray): True labels of shape ``(n_samples,)``.

        Returns:
            float: Mean accuracy in ``[0, 1]``.
        """
        return self.model.score(X, y)

    def get_params(self, deep=True):
        """Get parameters of the underlying estimator.

        Args:
            deep (bool): If ``True``, return the parameters of nested
                sub-objects too.

        Returns:
            dict: Mapping of parameter names to their values.
        """
        return self.model.get_params(deep)

    def set_params(self, **params):
        """Set parameters of the underlying estimator.

        Args:
            **params: Estimator parameters to set.

        Returns:
            RandomForestClassifier: The wrapped estimator with updated
            parameters.
        """
        return self.model.set_params(**params)
