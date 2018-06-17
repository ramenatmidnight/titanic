from sklearn.model_selection import train_test_split


def split_train_test(X, y):
    """
    Handle splitting
    :param X:
    :param y:
    :return:
    """
    print("> Splitting train test set...")

    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.2)

    print("Train set: %s" % str(X_train.shape))
    print("Test set: %s" % str(X_eval.shape))
    print("----------------")
    return X_train, X_eval, y_train, y_eval
