import sys
import pandas as pd
from exception import CustomException
from utils import load_object

class predict_pipeline():
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(model_path)
            preprocessor=load_object(preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
             raise CustomException(e,sys)

class Custom_data:
    def __init__(self,
                 fixed_acidity: float,
                 volatile_acidity: float,
                 citric_acid: float,  # ✅ Ensure it's a float
                 residual_sugar: float,
                 chlorides: float,
                 free_sulfur_dioxide: float,
                 total_sulfur_dioxide: float,
                 density: float,
                 pH: float,
                 sulphates: float,
                 alcohol: float):
        self.fixed_acidity = fixed_acidity
        self.volatile_acidity = volatile_acidity
        self.citric_acid = citric_acid  
        self.residual_sugar = residual_sugar
        self.chlorides = chlorides
        self.free_sulfur_dioxide = free_sulfur_dioxide
        self.total_sulfur_dioxide = total_sulfur_dioxide
        self.density = density
        self.pH = pH
        self.sulphates = sulphates
        self.alcohol = alcohol


    def get_data_as_frame(self):
        try:
            custom_data_point_dict={
                   
            "fixed_acidity" :[self.fixed_acidity],
            "volatile_acidity":[self.volatile_acidity],
            "citric_acid":[self.citric_acid] ,
            "residual_sugar":[self.residual_sugar],
            "chlorides":[self.chlorides], 
            "free_sulfur_dioxide":[self.free_sulfur_dioxide], 
            "total_sulfur_dioxide":[self.total_sulfur_dioxide], 
            "density":[self.density],
            "pH":[self.pH],
            "sulphates":[self.sulphates],
            "alcohol":[self.alcohol]
            }
            return pd.DataFrame(custom_data_point_dict)
        except Exception as e:
             raise CustomException(e,sys)




              