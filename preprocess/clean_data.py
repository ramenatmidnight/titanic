def clean_data(df):
    """
    Deal with NaNs, null
    :param df:
    :return:
    """
    print("> Cleaning dataframe...")
    print("Shape before: %s" % str(df.shape))

    cols_to_check = [
        "Pclass",
        "Sex",
        "Age",
        "Fare"
    ]

    print(df[cols_to_check].isnull().sum())
    df = remove_na(df, cols_to_check)
    print("Shape after: %s" % str(df.shape))
    print("----------------")
    return df


def remove_na(df, colnames):
    print(">> Removing NA from %s" % colnames)
    return df.dropna(subset=colnames)
