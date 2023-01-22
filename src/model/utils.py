

def decode(df, user_dict, movie_dict):
    # decode users and movies
    user_dict_decoder = {val: key for key, val in user_dict.items()}
    movie_dict_decoder = {val: key for key, val in movie_dict.items()}
    df_cols = df.columns
    if "User" in df_cols:
        df["User"] = df["User"].agg(lambda x: user_dict_decoder[x])
    if "Title" in df_cols:
        df["Title"] = df["Title"].agg(lambda x: movie_dict_decoder[x])
    return df
