import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from utils.setup import setup_paths


def explore_data(df):
    paths = setup_paths()
    print(df.info())
    print(df.head(2))

    dict_col_by_class = {
        1: "gold",
        2: "silver",
        3: "brown"
    }
    dict_col_by_sex = {
        "male": "blue",
        "female": "pink"
    }
    dict_col_by_sur = {
        0: "red",
        1: "green"
    }

    mpl.rcParams['toolbar'] = 'None'
    fig1 = plt.figure(figsize=(12, 10))

    plt.subplot2grid((2, 3), (0, 0))
    df_surv = df.Survived
    df_surv.value_counts(normalize=True).plot(
        alpha=0.5, kind='bar',
        color=df_surv.apply(lambda x: dict_col_by_sur.get(x)).unique())
    plt.title("Survived count")

    plt.subplot2grid((2, 3), (0, 1))
    df_pclass = df.Pclass
    df.Pclass.value_counts(normalize=True).plot(
        alpha=0.5, kind='bar',
        color=df_pclass.apply(lambda x: dict_col_by_class.get(x)).unique())
    plt.title("Pclass count")

    plt.subplot2grid((2, 3), (0, 2))
    plt.scatter(df.Survived, df.Age, alpha=0.02)
    plt.title("Age wrt Survived")

    plt.subplot2grid((2, 3), (1, 0), colspan=2)
    for p_class in df.Pclass.unique():
        df.Age[df.Pclass == p_class].plot(
            kind='kde',
            color=dict_col_by_class.get(p_class))
    plt.title("Class wrt Age")
    plt.legend(df.Pclass.unique())

    plt.subplot2grid((2, 3), (1, 2))
    df_sex = df.Sex
    df_sex.value_counts(normalize=True).plot(
        alpha=0.5, kind='bar',
        color=df_sex.apply(lambda x: dict_col_by_sex.get(x)).unique())
    plt.title("Sex count")

    plt.interactive(False)
    plt.tight_layout()
    fig1.savefig(paths.get("output_plots_path") + "fig1")

    ##################

    fig2 = plt.figure(figsize=(12, 10))

    plt.subplot2grid((3, 4), (0, 0))
    df_surv = df.Survived
    df_surv.value_counts(normalize=True).plot(
        alpha=0.5, kind='bar',
        color=df_surv.apply(lambda x: dict_col_by_sur.get(x)).unique())
    plt.title("Survived count")

    plt.subplot2grid((3, 4), (0, 1))
    df_sex_by_surv = df.Sex[df.Survived == 1]
    df_sex_by_surv.value_counts(normalize=True).plot(
        alpha=0.5, kind='bar',
        color=df_sex_by_surv.apply(lambda x: dict_col_by_sex.get(x)).unique())
    plt.title("Survived Sex count")

    _col = 2
    for _sex in df.Sex.unique():
        plt.subplot2grid((3, 4), (0, _col))
        df_surv_by_sex = df.Survived[df.Sex == _sex]
        df_surv_by_sex.value_counts(
            normalize=True
        ).plot(
            alpha=0.5, kind='bar',
            color=df_surv_by_sex.apply(lambda x: dict_col_by_sur.get(x)).unique()
        )
        _col += 1
        plt.title("Survived wrt Sex: %s" % _sex)

    plt.subplot2grid((3, 4), (1, 0), colspan=4)
    for p_class in df.Pclass.unique():
        df.Survived[df.Pclass == p_class].plot(
            kind='kde',
            color=dict_col_by_class.get(p_class))
    plt.title("Class wrt Survival")
    plt.legend(df.Pclass.unique())

    plt.subplot2grid((3, 4), (2, 0))
    df_poor_male = df.Survived[(df.Pclass == 3) & (df.Sex == "male")]
    df_poor_male.value_counts(normalize=True).plot(
        alpha=0.5, kind='bar',
        color=df_poor_male.apply(lambda x: dict_col_by_sur.get(x)).unique())
    plt.title("Poor male survival")

    plt.subplot2grid((3, 4), (2, 1))
    df_rich_male = df.Survived[(df.Pclass == 1) & (df.Sex == "male")]
    df_rich_male.value_counts(normalize=True).plot(
        alpha=0.5, kind='bar',
        color=df_rich_male.apply(lambda x: dict_col_by_sur.get(x)).unique())
    plt.title("Rich male survival")

    plt.subplot2grid((3, 4), (2, 2))
    df_poor_female = df.Survived[(df.Pclass == 3) & (df.Sex == "female")]
    df_poor_female.value_counts(normalize=True).plot(
        alpha=0.5, kind='bar',
        color=df_poor_female.apply(lambda x: dict_col_by_sur.get(x)).unique())
    plt.title("Poor female survival")

    plt.subplot2grid((3, 4), (2, 3))
    df_rich_female = df.Survived[(df.Pclass == 1) & (df.Sex == "female")]
    df_rich_female.value_counts(normalize=True).plot(
        alpha=0.5, kind='bar',
        color=df_rich_female.apply(lambda x: dict_col_by_sur.get(x)).unique())
    plt.title("Rich female survival")

    plt.interactive(False)
    plt.tight_layout()
    fig2.savefig(paths.get("output_plots_path") + "fig2")

    # ##################
    # fig3 = plt.figure(figsize=(12, 10))
    # arr_survived = []
    # arr_dead = []
    # df_extracted = df.Survived[(df.Pclass == 1) & (df.Sex == "male")] \
    #     .value_counts(normalize=True)
    # arr_survived.append(df_extracted[0])
    # arr_dead.append(df_extracted[1])
    # df_extracted = df.Survived[(df.Pclass == 2) & (df.Sex == "male")] \
    #     .value_counts(normalize=True)
    # arr_survived.append(df_extracted[0])
    # arr_dead.append(df_extracted[1])
    # df_extracted = df.Survived[(df.Pclass == 3) & (df.Sex == "male")] \
    #     .value_counts(normalize=True)
    # arr_survived.append(df_extracted[0])
    # arr_dead.append(df_extracted[1])
    # df_extracted = df.Survived[(df.Pclass == 1) & (df.Sex == "female")] \
    #     .value_counts(normalize=True)
    # arr_survived.append(df_extracted[0])
    # arr_dead.append(df_extracted[1])
    # df_extracted = df.Survived[(df.Pclass == 2) & (df.Sex == "female")] \
    #     .value_counts(normalize=True)
    # arr_survived.append(df_extracted[0])
    # arr_dead.append(df_extracted[1])
    # df_extracted = df.Survived[(df.Pclass == 3) & (df.Sex == "female")] \
    #     .value_counts(normalize=True)
    # arr_survived.append(df_extracted[0])
    # arr_dead.append(df_extracted[1])
    #
    # print(arr_survived)
    # print(arr_dead)
    # ind = np.arange(len(arr_survived))
    # plt.bar(ind, arr_survived, color='green')
    # plt.bar(ind, arr_dead, bottom=arr_survived, color='red')
    #
    # plt.interactive(False)
    # plt.tight_layout()
    # fig3.savefig(paths.get("output_plots_path") + "fig3")
