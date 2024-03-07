from src.mlproject_1.logger import logging
from src.mlproject_1.exception import CustomException
from src.mlproject_1.components.data_transformation import DataTransformation
import sys

from src.mlproject_1.components.data_ingestion import DataIngestion



if __name__ == '__main__':
    logging.info('Custome Exception ')
    try:
        # data ingestion
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiat_data_ingestion()

        # data transformation
        data_transformation = DataTransformation()
        train_arr , test_arr,_ = data_transformation.initiat_data_transformation(train_path, test_path)
        
    except Exception as e:
        raise CustomException(e, sys)