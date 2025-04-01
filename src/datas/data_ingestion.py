import pandas as pd 
import numpy as np 
import logging
import os

from sklearn.model_selection import train_test_split
import yaml


# configure logging
logger = logging.getLogger('Data_ingestion_log')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('logging.log')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_params(parameter):
    logger.debug('loading parameters file')
    try:
        with open('params.yaml','r') as file:
            file = yaml.safe_load(file)
            parameter = file['Data_ingestion'][parameter]
            return parameter
        logger.debug(f"{parameter} successfully fetched")
    except Exception as e:
        logger.error(f"{e}")

def load_data(path:str)->pd.DataFrame:
    try:
        logger.debug(f"loading data from {path}")
        df = pd.read_csv(path)
        logger.debug(f"data loaded successfully")
        return df
    except Exception as e:
        logger.error(f"loading data unsuccessfull -> {e}")
        print(f"loading data unsuccessfull -> {e}")

def create_target(data:pd.DataFrame)->pd.DataFrame:
    try:
        logger.debug('create binary target column as claim')
        data['claim'] = (data['CLM_AMT']==0).astype('int')
        logger.debug('binary target columns created as claim')
        logger.debug('droping CLM_AMT regression target column')
        data = data.drop(columns=['CLM_AMT'])
        logger.debug(' CLM_AMT column dropped')
        logger.debug('create target executed sucessfully')
        return data
    except Exception as e:
        logger.error(f"{e}")

def tran_test_split(data:pd.DataFrame,test_size:float)->tuple[pd.DataFrame]:
    try:
        logger.debug('creating training and testing set')
        X_train,X_test = train_test_split(data,test_size=test_size)
        logger.debug('training and testing set created sucessfully')
        return X_train,X_test
        logger.debug('Function train_test_split executed successfully')
    except Exception as e:
        logger.error(f"{e}")

def save_data(path:str,training_data:pd.DataFrame,testing_data:pd.DataFrame)->None:
    try:
       
        
        logger.debug(f"creating a directiory as {path}")
        os.makedirs(path,exist_ok=True)
        logger.debug('directory created sucessfully')
        logger.debug(f"saving data on {path}")
        training_data.to_csv(os.path.join(path,'train.csv'))
        testing_data.to_csv(os.path.join(path,'test.csv'))
        logger.debug(f"data save successfully on path {path}")
    except Exception as e:
        logger.error(f"{e}")

# main 
def main():
    try:
        data_path = "data\external\data_clean_l1.xls"
        data = load_data(data_path)
        data = create_target(data)
        test_size = load_params('test_size')
        X_train,X_test = tran_test_split(data,test_size)
        save_data_path = os.path.join('data','raw')
        save_data(save_data_path,X_train,X_test)
        logger.debug('Data ingestion successfully completed')
    except Exception as e:
        logger.error('Data Ingestion Failed')
        logger.error(f"{e}")

if __name__=="__main__":
    main()
