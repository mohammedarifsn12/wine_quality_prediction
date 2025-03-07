import os
import sys
from pathlib import Path
from exception import CustomException
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from logger import logger
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer
from components.model_trainer import ModelTrainerConfig
@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info('Entered the data ingestion method')
        try:
            df=pd.read_csv('notebooks/data/winequality-red.csv')
            print("Column names in the DataFrame:", df.columns.tolist())
            df.columns = df.columns.str.replace(" ", "_")  # Replace spaces with underscores
            df.to_csv("cleaned_dataset.csv", index=False)  # Save the cleaned dataset   
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            train_set,test_set=train_test_split(df,test_size=0.20,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logger.info('ingestion of data is completed')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    ingestion_obj=DataIngestion()
    train_data,test_data=ingestion_obj.initiate_data_ingestion()
    transformation_obj=DataTransformation()
    train_arr,test_arr,_=transformation_obj.initiate_data(train_data,test_data)
    trainer_obj=ModelTrainer()
    print(trainer_obj.initiate_model_trainer(train_arr,test_arr))
