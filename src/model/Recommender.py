from typing import Union

import pandas as pd

from model.BaseRecommender import BaseRecommender
from model.CBRecommender import CBRecommender
from model.CFRecommender import CFRecommender
from model.UCSRecommender import UCSRecommender


class Recommender_trivial(BaseRecommender):
    """Base abstract class for recommendation techniques."""

    def __init__(self, df_user: pd.DataFrame, df_movie: pd.DataFrame):
        super().__init__(df_user, df_movie)
        self.CB = CBRecommender(df_movie)
        self.UCS = UCSRecommender(df_user, df_movie)
        self.CF = CFRecommender(df_user, df_movie)

    def train(self, df_user: Union[pd.DataFrame, None], **kwargs):
        # self.CB.train(df_user, **kwargs)
        self.UCS.train(df_user, **kwargs)
        self.CF.train(df_user, **kwargs)

    def predict(self, df_user: pd.DataFrame, num_of_recomendations: int, **kwargs):
        little = 10

        UCS_movies = (2 * num_of_recomendations) // 10
        CB_movies = (4 * num_of_recomendations) // 10
        CF_movies = num_of_recomendations - CB_movies - UCS_movies

        recommendations = pd.DataFrame(columns=["User", "Title"])
        for user in df_user["User"].unique():
            if df_user[df_user["User"] == user].count() < little:
                recommendations_for_user = self.USC.predict(
                    df_user[df_user["User"] == user], num_of_recomendations
                )
            else:
                recommendations_for_user = pd.concat(
                    [
                        self.UCS.predict(df_user[df_user["User"] == user], UCS_movies),
                        self.CB.predict(df_user[df_user["User"] == user], CB_movies),
                        self.CF.predict(df_user[df_user["User"] == user], CF_movies),
                    ]
                )
            recommendations = pd.concat([recommendations, recommendations_for_user])
        return recommendations


class Recommender(BaseRecommender):
    """Base abstract class for recommendation techniques."""

    def __init__(self, df_movie: pd.DataFrame):
        super().__init__(df_movie)
        self.CB = CBRecommender(df_movie)
        self.UCS = UCSRecommender(df_movie)
        self.CF = CFRecommender(df_movie)

    def train(self, df_user: Union[pd.DataFrame, None], **kwargs):
        self.UCS.train(df_user, **kwargs)
        self.CF.train(df_user, **kwargs)

    def predict(self, df_user: pd.DataFrame, num_of_recomendations: int, **kwargs):
        little = kwargs["little"]
        CB_weight = kwargs["CB_weight"]
        UCS_weight = kwargs["UCS_weight"]
        CF_weight = kwargs["CF_weight"]
        if little is None:
            little = 10
        if CB_weight is None:
            CB_weight = 0.27415096
        if UCS_weight is None:
            UCS_weight = 0.10017023
        if CF_weight is None:
            CF_weight = 1 - CB_weight - UCS_weight
        assert 0 <= CB_weight and CB_weight <= 1
        assert 0 <= UCS_weight and UCS_weight <= 1
        assert 0 <= CF_weight and CF_weight <= 1
        recommendations = pd.DataFrame(columns=["User", "Title"])
        for user in df_user["User"].unique():
            if len(df_user[df_user["User"] == user]) < little:
                recommendations_for_user = self.UCS.predict(
                    df_user[df_user["User"] == user], num_of_recomendations
                )
            else:
                recommendations_CB = self.CB.predict(
                    df_user[df_user["User"] == user], -1
                )
                recommendations_UCS = self.UCS.predict(
                    df_user[df_user["User"] == user], -1
                )
                recommendations_CF = self.CF.predict(
                    df_user[df_user["User"] == user], -1
                )
                recommendations_for_user = recommendations_CB.copy()
                recommendations_for_user["Rating"] = (
                    (recommendations_CB["Rating"] * CB_weight)
                    .add(
                        recommendations_UCS["Rating"] * UCS_weight,
                        fill_value=0,
                    )
                    .add(
                        recommendations_CF["Rating"] * CF_weight,
                        fill_value=0,
                    )
                )
            recommendations_for_user = recommendations_for_user.sort_values(
                "Rating", ascending=False
            ).head(num_of_recomendations)

            recommendations = pd.concat([recommendations, recommendations_for_user])
        return recommendations
