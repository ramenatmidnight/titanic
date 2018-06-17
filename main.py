from preprocess.explore_data import explore_data
from preprocess.clean_data import clean_data
from preprocess.extract_features import extract_features
from preprocess.load_data import load_data
from utils.setup import setup_paths


def main():
    print("HELLO")

    paths = setup_paths()
    raw_data_path = paths.get("raw_data_path") + "train.csv"

    df = load_data(raw_data_path)

    df = clean_data(df)

    explore_data(df)
    # Age, Sex, Pclass seems to contribute to Survived

    X_train, y_train = extract_features(df)


if __name__ == '__main__':
    main()
