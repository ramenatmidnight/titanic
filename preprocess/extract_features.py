def extract_features(df):
    """
    Get the
    :param df:
    :return:
    """
    print("> Extracting features...")

    # features that will be used
    features = [
        # "PassengerId",  # only needed for submission
        # "Survival",  # this is our Target
        "Pclass",
        # "Name",  # can this be relevant?
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        # "Ticket",  # the ID of the ticket
        "Fare",
        # "Cabin",  # skip this, too many NAs
        "Embarked"
    ]
    X = df[features].values

    # the "target"
    y = []
    if "Survived" in df.columns:
        y = df["Survived"].values

    print("Features: %s" % features)
    print("Shape: %s" % str(X.shape))
    print("----------------")
    return X, y, features
