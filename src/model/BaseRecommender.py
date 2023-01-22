import pandas as pd
from typing import Union
from abc import ABC


class BaseRecommender(ABC):
    def __init__(self, df_user: pd.DataFrame, df_movie: pd.DataFrame):
        self.df_user = df_user
        self.df_movie = df_movie

    """ Base abstract class for recommendation techniques."""

    def train(self, df_user: Union[pd.DataFrame, None], **kwargs):
        raise NotImplementedError

    def predict(self, df_user: pd.DataFrame, num_of_recomendations: int, **kwargs):
        raise NotImplementedError
