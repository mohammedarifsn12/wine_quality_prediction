import os
import sys
import dill
import logger
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(x_train, y_train, x_test, y_test, models):                
    try:
        report = {}
        best_model_name = None
        best_model_score = float('-inf')  # Initialize with lowest score
        best_model = None
        for name, model in models.items():
            model.fit(x_train, y_train)  # Fix: Pass y_train during fitting

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            r2_score_train = r2_score(y_train, y_train_pred)
            r2_score_test = r2_score(y_test, y_test_pred)
            
            report[name] = r2_score_test 
            if r2_score_test>best_model_score:
                best_model_score=r2_score_test
                best_model_name=name
                best_model=model
        return best_model_score,best_model_name,best_model

        
    except Exception as e:
        raise CustomException(e, sys)


