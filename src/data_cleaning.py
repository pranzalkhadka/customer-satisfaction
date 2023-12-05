import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Union
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreProcessStrategy(DataStrategy):

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data = data.drop(
                [
                    #Certain Columns are removed to reduce the complexity of project
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis = 1,
            )
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
        except:
            pass