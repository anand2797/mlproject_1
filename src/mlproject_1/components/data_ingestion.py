import os
import sys
from src.mlproject_1.logger import logging
from src.mlproject_1.exception import CustomException
from src.mlproject_1.utils import read_sql_data

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass


# create path for data by using data classes and os module

@dataclass
class DataIngestionConfig:
    train_path:str = os.path.join('artifacts', 'train.csv')
    test_path:str = os.path.join('artifacts', 'test.csv')
    raw_path:str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiat_data_ingestion(self):
        try:
            # reading data from mysql
            df = pd.read_csv(os.path.join('artifacts', 'raw.csv'))
            logging.info('Reading from mysql database.')
            os.makedirs(os.path.dirname(self.ingestion_config.train_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_path, index=False, header=True)

            # train test split
            train_df , test_df = train_test_split(df, test_size=0.2, random_state=42)
            train_df.to_csv(self.ingestion_config.train_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_path, index=False, header=True)

            logging.info("Data ingestion is completed.")

            return (
                self.ingestion_config.train_path,
                self.ingestion_config.test_path
            )

        except Exception as e:
            raise CustomException(e, sys)




