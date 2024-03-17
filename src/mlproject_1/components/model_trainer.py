import os
import sys

from src.mlproject_1.logger import logging 
from src.mlproject_1.exception import CustomException
from src.mlproject_1.utils import save_object, evaluate_model


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    model_trainer_filepath = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiat_model_trainer(self, train_arr, test_arr):

        try:
            logging.info('Split train and test data as input and target data.')
            # split dependent and independent features from training and test data
            # train arrays
            X_train_arr = train_arr[:,:-1]
            y_train_arr = train_arr[:,-1]

            # test_arrays
            X_test_arr = test_arr[:,:-1]
            y_test_arr = test_arr[:,-1]

            # create a dictionary which contains machine learning algorithms
            models = {
                   "Linear Regression":LinearRegression(),
                   "Svm": SVC(),
                   "KNearest Neighbors": KNeighborsRegressor(),
                   "Decision Tree": DecisionTreeRegressor(),
                   "Random Forest": RandomForestRegressor()
            }
            # Create a dictionary with models and their respective parameters
            model_parameters = {
                "Linear Regression": {
                   # "fit_intercept": [True, False],  # Whether to calculate the intercept for this model
                   # "normalize": [True, False],  # Whether to normalize the features before regression
                },
                "Svm": {
                    "C": [0.1, 1, 10],  # Regularization parameter
                    "kernel": ['linear', 'rbf'],  # Kernel type
                    "gamma": ['scale', 'auto'],  # Kernel coefficient for 'rbf' kernel
                },
                "KNearest Neighbors": {
                    "n_neighbors": [3, 5, 7],  # Number of neighbors
                    "weights": ['uniform', 'distance'],  # Weight function used in prediction
                    "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm used to compute the nearest neighbors
                },
                "Decision Tree": {
                    "max_depth": [None, 10, 20],  # Maximum depth of the tree
                    "min_samples_split": [2, 5, 10],  # Minimum number of samples required to split an internal node
                    "min_samples_leaf": [1, 2, 4],  # Minimum number of samples required to be at a leaf node
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200],  # Number of trees in the forest
                    "max_depth": [None, 10, 20],  # Maximum depth of the tree
                    "min_samples_split": [2, 5, 10],  # Minimum number of samples required to split an internal node
                    "min_samples_leaf": [1, 2, 4],  # Minimum number of samples required to be at a leaf node
                }
            }

            model_report:dict = evaluate_model(X_train_arr, y_train_arr, X_test_arr, y_test_arr, models, model_parameters)
            # best model score from model_report
            best_model_score = max(list(model_report.values()))
            print(model_report)

            # get model index for model name in model_report
            model_index = list(model_report.values()).index(best_model_score)

            # model name 
            bet_model_name = list(model_report.keys())[model_index]

            # get best model 
            best_model = models[bet_model_name]

            if best_model_score < 0.6 :
                raise CustomException("No best model found")
            
            logging.info("Best model found on both train and test datasets.")

            save_object(
                file_path= self.model_trainer_config.model_trainer_filepath,
                obj= best_model
            )

            prediction = best_model.predict(X_test_arr)
            r2 = r2_score(y_test_arr, prediction)
            return f"model is {best_model} >>: score is {r2}"
            

        except Exception as e:
            raise CustomException(e, sys)

