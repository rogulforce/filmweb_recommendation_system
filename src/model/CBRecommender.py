import pandas as pd
import numpy as np
from typing import Union
from sklearn.preprocessing import MinMaxScaler

from model.BaseRecommender import BaseRecommender

class CBRecommender(BaseRecommender):
    def __init__(self, df_movie):
        self.movie_items = self.create_movie_items(df_movie)

    def create_movie_items(self, df_movie):
        """
        Function prepares movie_items matrix which:
        - In each row is single movie
        - In each column is movie's asset (mostly one hot encoding of actors and genres)
        """
        # Create columns assignment ('Gnere/Actor: {genre/actor}')
        # Its preparation for one hot encoding
        temp = df_movie.copy()
        temp["Genre_name"] = "Genre: " + temp["Genre"]
        temp["Actor_name"] = "Actor: " + temp["Actor"]

        # Create movie_items
        movie_items = (
            temp
            # returns Title and for different rows either actors and genre
            .melt("Title", ["Genre_name", "Actor_name"], value_name="Class")
            # Assign a ghost column with 1 which will turn to one hot encodng in pivot table
            .assign(value=1)
            # Sometimes for the same title occured multiple same genres... which occured after melting
            # For pivoting we have to discard them
            .drop_duplicates()
            # Pivot table => matrix of 0/1 with n_title rows x n_{actor/genre} columns
            .pivot(index="Title", columns="Class")["value"].fillna(0)
        )

        # We will concat the columns of average rating and number of ratings to movie_items
        temp2 = temp.groupby("Title").agg({"Avg_rating": min, "Number_of_ratings": min})

        # Because we will base our analysis on the value of dot product let's normalize new columns which
        # have values greater than 1
        scaler = MinMaxScaler()

        movie_items[["Avg_rating", "Number_of_ratings"]] = scaler.fit_transform(temp2)
        return movie_items

    def predict_for_single_user(self, df_user, user_id, n_recommendations):
        """Function recommends provided number of movies to user"""
        # Take user data
        user_data = df_user[df_user["User"] == user_id]

        # Find watched movies
        watched_movies = user_data.Title.values

        # Create user_items => select movie's items for user watched movies from our database
        user_items = self.movie_items.loc[watched_movies]

        # Take user diff ratings as an appropriate weight for dot product
        ratings = user_data["Avg_user_rating_diff"].values

        # Calculate dot product accross watched movies and those from db
        # Weight the solution by rating diff of user
        similarity_matrix = (self.movie_items @ user_items.T) * ratings
        # For watched movies assign value 0
        similarity_matrix.loc[watched_movies] = 0
        if n_recommendations == -1:
            n_recommendations = len(similarity_matrix)

        tmp = (similarity_matrix
            # Take max across row
            .max(axis=1)
            # Sort in descending way
            .sort_values(ascending=False)
        )
        title = (
            tmp
            # Take n recommendations
            .iloc[:n_recommendations]
            .index.values
            )
        rating = (tmp
            # Take n recommendations
            .iloc[:n_recommendations]
            .array.to_numpy()
            )

        for i in range(n_recommendations):
            yield title[i], rating[i]

    def train(
        self, df_user: Union[pd.DataFrame, None], n_top_movies: int = 20, **kwargs
    ):
        pass

    def predict(self, df_user: pd.DataFrame, num_of_recomendations: int, **kwargs):
        """Predict movie recomendations for provided users with their data"""
        # Find all unique users
        user_ids = np.unique(df_user["User"])

        # Generate recommendations
        recommendations = pd.DataFrame(
            [
                (user_id, movie, rate)
                for user_id in user_ids
                # Generate recommendations for each user
                for movie, rate in self.predict_for_single_user(
                    df_user, user_id, num_of_recomendations
                )
            ],
            columns=["User", "Title", "Rating"],
        )
        return recommendations