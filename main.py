from models.evaluate import evaluate_model
from models.train import train_model
from preprocess.extract_features import extract_features
from preprocess.load_data import load_data
from preprocess.preprocess_data import preprocess_data
from preprocess.split_train_test import split_train_test
from utils.setup import setup_paths


def main():
    paths = setup_paths()
    raw_data_path = paths.get("raw_data_path") + "train.csv"

    # get raw data, from /datasets/
    df = load_data(raw_data_path)

    # deal with cleaning/transforming raw data into the correct types
    df = preprocess_data(df)

    # building the relevant features set
    X, y, features = extract_features(df)
    X_train, X_eval, y_train, y_eval = split_train_test(X, y)

    model = train_model(X_train, y_train, features)
    evaluate_model(model, X_eval, y_eval)


if __name__ == '__main__':
    main()
