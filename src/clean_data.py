
def clean_data(df_user, df_movie):
    df_movie = df_movie[df_movie["Avg_rating"] <= 10]

    concat = lambda df, col1, col2: df[col1].astype(str) + "_" + df[col2].astype(str)

    df_movie["Title"] = concat(df_movie, "Title", "Year")
    df_user["Title"] = concat(df_user, "Title", "Year")
    df_user = df_user.drop(columns="Year")
    # percentage difference between user rating and average rating (user-avg)

    df_user_temp = df_user.merge(
        df_movie.groupby(["Title"])["Avg_rating"].max().reset_index(),
        on=["Title"],
        how="left",
    )

    df_user["Avg_user_rating_diff"] = df_user_temp["Rating"] - df_user_temp["Avg_rating"]

    df_user = df_user.dropna()
    # Mapping users to numbers
    user_dict = {user: i for i, user in enumerate(df_user.User.unique())}
    movie_dict = {title: i for i, title in enumerate(df_movie.Title.unique())}

    df_user[["User", "Title"]] = df_user[["User", "Title"]].agg(
        {"User": lambda x: user_dict[x], "Title": lambda x: movie_dict[x]}
    )
    df_movie[["Title"]] = df_movie[["Title"]].agg({"Title": lambda x: movie_dict[x]})
    return df_user, df_movie