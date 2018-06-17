from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train):
    """
    Train model
    :param X_train:
    :param y_train:
    :return:
    """
    print("> Building model...")
    model = LogisticRegression()
    print("Train set size: %s" % str(X_train.shape))
    model.fit(X_train, y_train)
    print("Model intercept: %s" % model.intercept_)
    print("Model coeffs: %s" % model.coef_)
    print("----------------")
    return model
