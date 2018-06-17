import os

import pandas as pd


def load_data(path):
    print("> Loading data from %s..." % str(os.path.basename(path)))
    df = pd.read_csv(path)
    print("Shape: %s" % str(df.shape))
    print("Columns: %s" % str(df.columns.values))
    print("----------------")
    return df
