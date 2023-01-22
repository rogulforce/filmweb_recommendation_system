import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd

DF2_DTYPES = {
    "Title": str,
    "Year": "int16",
    "Genre": str,
    "Actor": str,
    "Avg_rating": "float64",
    "Number_of_ratings": "int32",
}


def clean_and_save_data(
    df1_path: str = None, df2_path: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Method assumes that in data directory are pickle files with scraped
    data from filmweb.pl. It imports then, unpack and transform into two
    relevant data frames which are saved then to csv files.
    """
    data_dir_path = os.path.join("..", "data")
    if df1_path is None:
        df1_path = os.path.join(data_dir_path, "processed/df1.csv")
    if df2_path is None:
        df2_path = os.path.join(data_dir_path, "processed/df2.csv")

    # Valid files
    data_files_path = [
        file for file in os.listdir(data_dir_path) if file.endswith(".pickle")
    ]

    results_from_pickle = []
    for file_path in data_files_path:
        full_path = os.path.join(data_dir_path, file_path)
        with open(full_path, "rb") as f:
            temp = pickle.load(f)
            results_from_pickle.extend(temp)

    # Unpacking data to lists
    users, titles, years, genres, actors, avg_ratings, n_ratings, ratings = list(
        zip(
            *[
                (
                    sample[0],
                    sample[1],
                    sample[2],
                    sample[3][0],
                    sample[3][1],
                    sample[4],
                    sample[5],
                    sample[6],
                )
                for sample in results_from_pickle
            ]
        )
    )

    # Creating frame 1
    df1 = pd.DataFrame(
        {"User": users, "Title": titles, "Year": years, "Rating": ratings}
    )

    df1["Rating"] = (
        df1["Rating"].replace(" ", np.nan).replace("", np.nan).astype("float16")
    )

    df2 = pd.DataFrame(
        {
            "Title": titles,
            "Year": years,
            "Genre": genres,
            "Actor": actors,
            "Avg_rating": avg_ratings,
            "Number_of_ratings": n_ratings,
        }
    )

    # Creating frame 2 without duplicates
    df2 = (
        df2.explode("Genre").explode("Actor")
        # .drop_duplicates()
        # .reset_index(drop=True)
    )

    # Removing invalid behaviour of genre and actors
    invalid_genre = df2["Genre"].isna()
    df2.loc[invalid_genre, "Genre"] = df2.loc[invalid_genre, "Actor"]
    df2.loc[invalid_genre, "Actor"] = np.nan

    # Cleaning avg rating to proper float
    df2["Avg_rating"] = df2["Avg_rating"].str.replace(",", ".").str.replace(" ", "")
    df2 = df2.astype(DF2_DTYPES)

    # fix incorrect values from df2
    df2 = (
        df2.groupby(["Title", "Year", "Genre", "Actor"])[
            ["Avg_rating", "Number_of_ratings"]
        ]
        .agg(lambda x: x.mode()[0])
        .reset_index()
    )

    df1.to_csv(df1_path, index=False)
    df2.to_csv(df2_path, index=False)

    return df1, df2


def load_data(
    df1_path: str = None, df2_path: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Method loads csv files with processed data frames
    """
    data_dir_path = os.path.join("..", "data/processed")
    if df1_path is None:
        df1_path = os.path.join(data_dir_path, "df1.csv")
    if df2_path is None:
        df2_path = os.path.join(data_dir_path, "df2.csv")

    df1 = pd.read_csv(df1_path)
    df1["Rating"] = df1["Rating"].astype("float16")
    df2 = pd.read_csv(df2_path).astype(DF2_DTYPES)
    return df1, df2
