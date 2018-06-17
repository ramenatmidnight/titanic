from models.evaluate import evaluate_model
from models.train import train_model
from preprocess.explore_data import explore_data
from preprocess.clean_data import clean_data
from preprocess.extract_features import extract_features
from preprocess.load_data import load_data
from preprocess.split_train_test import split_train_test
from preprocess.transform_data import transform_data
from utils.setup import setup_paths


def main():
    paths = setup_paths()
    raw_data_path = paths.get("raw_data_path") + "train.csv"

    df = load_data(raw_data_path)

    # explore_data(df)
    # # Age, Sex, Pclass seems to contribute to Survived

    df = transform_data(df)
    df = clean_data(df)

    X, y = extract_features(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)


if __name__ == '__main__':
    main()
