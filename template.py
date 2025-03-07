from pathlib import Path
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# Define the list of required files
list_of_files = [
    "src/__init__.py",
    "src/logger.py",
    "src/exception.py",
    "src/utils.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/pipeline/__init__.py",
    "src/pipeline/predict_pipeline.py",  # Added .py extension
    "templates/index.html",
    "templates/home.html",
    "application.py",
    "setup.py",
    "requirements.txt"
]


for file in list_of_files:
    file_path = Path(file)
    file_dir, file_name = os.path.split(file_path)

    
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for the file: {file_name}")

   
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path, 'w') as f:  
            pass  
        logging.info(f"Creating empty file: {file_path}")
    else:
        logging.info(f"{file_name} already exists")
