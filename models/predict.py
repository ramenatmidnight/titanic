from preprocess.extract_features import extract_features
from preprocess.load_data import load_data
from preprocess.transform_data import transform_data


def predict_for_items(model, X):
    """
    :return:
    """
    print(">> Predicting items...")
    y_predicted = model.predict(X)
    print(y_predicted)
    print("----------------")
    return y_predicted


def predict_for_csv(model, test_data_path):
    """
    :return:
    """
    print("> Predicting for csv file...")
    df_test = load_data(test_data_path)
    df_test = transform_data(df_test)
    X_test, _ = extract_features(df_test)
    print("Predicting set: %s" % str(X_test.shape))
    print("----------------")
    return predict_for_items(model, X_test)
