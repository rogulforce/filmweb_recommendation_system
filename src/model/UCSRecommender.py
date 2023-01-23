import pandas as pd
from typing import Union

from model.BaseRecommender import BaseRecommender

class UCSRecommender(BaseRecommender):
    """user cold start recommender. (For users with not many rated movies.)
    Take n_top_movies and recommend them in order based on Avg_rating."""

    recommendation_table = None

    def train(
        self, df_user: Union[pd.DataFrame, None], n_top_movies: int = 20, **kwargs
    ):
        self.recommendation_table = (
            self.df_movie.groupby(["Title"])[["Avg_rating", "Number_of_ratings"]]
            .min()
            .reset_index()
            .sort_values("Number_of_ratings", ascending=False)
            .head(n_top_movies)
            .sort_values("Avg_rating", ascending=False)
        )

    def predict(self, df_user: pd.DataFrame, num_of_recomendations: int, **kwargs):
        recommendations = pd.DataFrame(columns=["User", "Title"])
        for user in df_user["User"].unique():
            recommendations_for_user = self.recommendation_table.head(
                num_of_recomendations
            )[["Title", "Avg_rating"]].rename(columns={"Avg_rating":"Rating"})
            recommendations_for_user["User"] = user
            recommendations = pd.concat([recommendations, recommendations_for_user])
        return recommendations