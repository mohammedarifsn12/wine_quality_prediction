from ast import List
from pathlib import Path
import os 
from setuptools import setup,find_packages
HYPHEN_E_DOT = "-e ."
def get_requirements(file_name:str)-> List :
    with open(file_name) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.strip() for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

            return requirements

setup(
    name='Wine Quality prediction',
    version='0.0.0',
    author='Mohammed Arif S N',
    author_email='mohammedarifsn2@gmail.com',
    description='Creating an website which predicts quality of an wine',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=get_requirements('requirements.txt')
)