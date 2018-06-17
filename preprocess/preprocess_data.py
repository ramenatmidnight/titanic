from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):
    """
    Stuff
    :param df:
    :return:
    """
    print("> Preprocessing...")

    # transform categorical data to numeric
    df = transform_data(df)
    # deal with nans and bad data
    df = clean_data(df)

    # Done with the obvious, now do other things
    df.Embarked = LabelEncoder().fit_transform(df.Embarked.fillna("S"))

    print("----------------")
    return df


def transform_data(df):
    """
    Get the obvious correct types
    :param df:
    :return:
    """
    print(">> Transforming dataframe...")
    # encode "male" and "female"
    df.Sex = LabelEncoder().fit_transform(df.Sex)
    print("-----------")
    return df


def clean_data(df):
    """
    Deal with obvious NaNs, null
    :param df:
    :return:
    """
    print(">> Cleaning dataframe...")
    print("Shape before: %s" % str(df.shape))
    print("Null cells:")
    print(df.isnull().sum())

    # fill these NA columns
    cols_to_fill = [
        "Pclass",
        "Sex",
        "Age",
        "Fare"
    ]
    df = fill_na_mean(df, cols_to_fill)

    # remove these NA columns
    cols_to_remove = []
    df = remove_na(df, cols_to_remove)

    print("Shape after: %s" % str(df.shape))
    print("-----------")
    return df


def remove_na(df, colnames):
    print(">>> Removing NA from %s" % colnames)
    return df.dropna(subset=colnames)


def fill_na_mean(df, colnames):
    print(">>> Filling NA with mean for %s" % colnames)
    for col in colnames:
        df[col] = df[col].fillna(df[col].dropna().mean())
    return df


def fill_na_median(df, colnames):
    print(">>> Filling NA with mean for %s" % colnames)
    for col in colnames:
        df[col] = df[col].fillna(df[col].dropna().median())
    return df
