from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train, features=None):
    """
    Train model
    :param features:
    :param X_train:
    :param y_train:
    :return:
    """
    print("> Building model...")
    model = LogisticRegression()
    print("Train set size: %s" % str(X_train.shape))
    model.fit(X_train, y_train)

    print("Model intercept: %s" % model.intercept_)
    if features is not None:
        for f, c in zip(features, model.coef_.flatten()):
            print("* %s: %s" % (f, c))
    else:
        print("Model coeffs: %s" % model.coef_.flatten())
    print("----------------")
    return model
