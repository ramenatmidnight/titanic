from explore.explore_data import explore_data
from preprocess.load_data import load_data
from utils.setup import setup_paths


def main():
    paths = setup_paths()
    raw_data_path = paths.get("raw_data_path") + "train.csv"

    df = load_data(raw_data_path)
    explore_data(df)


if __name__ == '__main__':
    main()
