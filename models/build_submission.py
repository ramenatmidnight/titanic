from preprocess.extract_features import extract_features
from preprocess.load_data import load_data
from preprocess.transform_data import transform_data


def build_submission(model, test_data_path, baseline_path):
    print("> Building kaggle submission...")
    df_test = load_data(test_data_path)
    df_test = transform_data(df_test)
    X_test, _ = extract_features(df_test)

    df_baseline = load_data(baseline_path)
    # print(df_baseline)

    # TODO: for NaNs, we just use the "baseline" predictor?

    y_predicted = model.predict(X_test)
    print(y_predicted)
    print("----------------")

