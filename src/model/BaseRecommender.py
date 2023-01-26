from abc import ABC
from typing import Union

import pandas as pd


class BaseRecommender(ABC):
    def __init__(self, df_movie: pd.DataFrame):
        self.df_movie = df_movie

    """ Base abstract class for recommendation techniques."""

    def train(self, df_user: Union[pd.DataFrame, None], **kwargs):
        raise NotImplementedError

    def predict(self, df_user: pd.DataFrame, num_of_recomendations: int, **kwargs):
        raise NotImplementedError
