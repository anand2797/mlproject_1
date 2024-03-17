import os
import sys
from src.mlproject_1.exception import CustomException
from src.mlproject_1.logger import logging 

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

import pandas as pd 
from dotenv import load_dotenv
import pymysql
import pickle

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")



def read_sql_data():
    logging.info("Reading SQL database started.")
    try:
        mydb = pymysql.connect(
            host = host,
            user = user,
            password = password,
            db = db
        )
        logging.info("Connection Established with mysql database",mydb)

        df = pd.read_sql("select * from students", mydb)
        print(df)

        return df
    except Exception as ex:
        raise CustomException(ex)
    

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)  
    except Exception as e:
        raise CustomException(e,sys)  

def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            params = param[list(models.keys())[i]]

            grid_cv = GridSearchCV(model, params, cv=5)
            grid_cv.fit(X_train, y_train)
            
            model.set_params(**grid_cv.best_params_)
            model.fit(X_train, y_train)

            # check predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # r2 score check
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            # store result in report dictionary for each model
            report[list(models.keys())[i]] = test_score

        return report 
    
    except Exception as e:
        raise CustomException(e, sys)
    







     
 
 