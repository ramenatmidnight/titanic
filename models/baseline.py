from sklearn.linear_model import LogisticRegression


class BaselineGender(LogisticRegression):
    """
    This model simply predict that all Female passenger survived
    """
    def predict(self, X):
        sex = X[:, 0]
        return 1 - sex
