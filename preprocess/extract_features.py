def extract_features(df):
    """
    Get the
    :param df:
    :return:
    """
    print("> Extracting features...")

    # features that will be used
    X = df[[
        "Sex",
        "Pclass",
        "Age",
        "Fare"
    ]].values

    # the "target"
    y = []
    if "Survived" in df.columns:
        y = df["Survived"].values

    print("Shape: %s" % str(X.shape))
    print("----------------")
    return X, y
