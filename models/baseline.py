from sklearn.linear_model import LogisticRegression


class BaselineGender(LogisticRegression):
    """
    This model simply predict that all Female passenger survived
    """
    def predict(self, X):
        """
        Hardcoded position 1 for Sex column
        :param X:
        :return:
        """
        sex = X[:, 1]
        return 1 - sex
