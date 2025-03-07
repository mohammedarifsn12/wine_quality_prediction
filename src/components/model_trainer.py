import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from exception import CustomException
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from utils import evaluate_model, save_object
from logger import logger

@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logger.info('Starting model training')

            # Fix train-test split
            x_train, y_train = train_array[:, :-1], train_array[:, -1]
            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "KNN": KNeighborsRegressor()
            }

            best_model_score,best_model_name,best_model = evaluate_model(x_train, y_train, x_test, y_test, models)

            
            
            logger.info(f"Best Model: {best_model_name} with R² Score: {best_model_score}")
            save_object(self.model_trainer_config.trained_model_path,best_model)


            predicted=best_model.predict(x_test)
            r2_sc=r2_score(y_test,predicted)

            return r2_sc
        except Exception as e:
            raise CustomException(e, sys)
