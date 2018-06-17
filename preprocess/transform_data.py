from sklearn.preprocessing import LabelEncoder


def transform_data(df):
    """
    Get the correct types
    :param df:
    :return:
    """
    print("> Transforming dataframe...")
    le = LabelEncoder()
    df.Sex = le.fit_transform(df.Sex)
    print("----------------")
    return df
