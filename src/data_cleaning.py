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
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)

            return data
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e
        

class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into train and test
    """
    def handle_data(self, data: pd.DataFrame) -> Union([pd.DataFrame, pd.Series]):
        try:
            X = data.drop(["review_score"], axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e
        
#This class can run any strategy defined above like DataPreProcessStrategy or DataDivideStrategy
#This is called strategy design pattern
class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_data(self.df)