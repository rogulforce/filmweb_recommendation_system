from typing import Union

import pandas as pd
from surprise import SVD, Dataset, KNNWithMeans, Reader, accuracy

from model.BaseRecommender import BaseRecommender


class CFRecommender(BaseRecommender):
    def __init__(self, df_user, df_movie):
        super(CFRecommender, self).__init__(df_user, df_movie)
        self.algo_SVD = SVD()
        # rating scale
        # reader = Reader(rating_scale = (1,10))
        # self.rating_df = Dataset.load_from_df(df_user[['User','Title', 'Rating']], reader)

    def train(
        self, df_user: Union[pd.DataFrame, None], n_top_movies: int = 20, **kwargs
    ):
        reader = Reader(rating_scale=(1, 10))
        self.rating_df = Dataset.load_from_df(
            df_user[["User", "Title", "Rating"]], reader
        )

        self.algo_SVD.fit(self.rating_df.build_full_trainset())

    def predict(self, df_user: pd.DataFrame, num_of_recomendations: int, **kwargs):
        recommendations = pd.DataFrame(columns=["User", "Title", "Rating"])
        for user in df_user["User"].unique():

            user_movies = df_user[df_user["User"] == user]["Title"].unique()

            # predict value for each movie in dataset.
            pred_list = []
            for movie in range(1, len(self.df_movie.Title.unique())):
                rating = self.algo_SVD.predict(user, movie).est
                pred_list.append([user, movie, rating])

            recommendations_for_user = pd.DataFrame(
                pred_list, columns=["User", "Title", "Rating"]
            )

            # remove already watched movies from recommendations
            recommendations_for_user = (
                recommendations_for_user[
                    ~recommendations_for_user["Title"].isin(user_movies)
                ]
                .sort_values("Rating", ascending=False)
                .head(num_of_recomendations)
            )
            recommendations = pd.concat([recommendations, recommendations_for_user])
        return recommendations
