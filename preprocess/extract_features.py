def extract_features(df):
    """
    Get the
    :param df:
    :return:
    """
    return df[[
        "Pclass",
        "Sex",
        "Age",
        # "Fare"
    ]].values, df["Survived"].values
