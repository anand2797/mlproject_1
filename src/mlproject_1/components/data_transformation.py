import sys
from src.mlproject_1.exception import CustomException
from src.mlproject_1.logger import logging
from src.mlproject_1.utils import save_object

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

import os
import pandas as pd
import numpy as np

from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_obj_filepath = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self, num_cols, cat_cols):
        '''
        This fuction will do data transformation.
        '''
        try:
            
           
            """numerical_featurs = ["writing_score", "reading_score"]
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]"""
            
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler(with_mean=False))
                 ])
            
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info(f'Categorical columns: {cat_cols}')
            logging.info(f'Numerical columns: {num_cols}')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_cols),
                    ('cat_pipeline', cat_pipeline, cat_cols) 
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        


    def initiat_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading train and test data file.')

            
           # mention columns 
            target_column="math_score"
            
            numerical_features = ["writing_score", "reading_score"]
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # preprocessing object which perform preprocessing task
            preprocessing_obj = self.get_data_transformation_obj(numerical_features, categorical_features)

            # split dependent and independent feature for training data
            input_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            # split dependent and independent feature for test data
            input_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            # applying preprocessing on trainig and test data
            input_train_array = preprocessing_obj.fit_transform(input_train_df)
            input_test_array = preprocessing_obj.transform(input_test_df)

            # concatenate arrays
            train_array = np.c_[input_train_array, np.array(target_feature_train_df)]

            test_array = np.c_[input_test_array, np.array(target_feature_test_df)]

            logging.info('Completely saved preprocessing object.')

            # save preprocessor in pkl file by using utils.py --> save_object fuction
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_filepath,
                obj= preprocessing_obj
            )
            return (train_array, 
                    test_array, 
                    self.data_transformation_config.preprocessor_obj_filepath
                    )
        
        except Exception as e:
            raise CustomException(e, sys)



