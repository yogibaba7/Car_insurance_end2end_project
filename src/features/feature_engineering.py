import pandas as pd 
import numpy as np 
import logging 
import os 
from sklearn.preprocessing import OrdinalEncoder
import joblib

# configure logging
logger = logging.getLogger('Feature_engineering_log')
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
            parameter = file['Data_ingestion'][parameter]  # there should be feature engineering params at this time i leave it .
            return parameter
        logger.debug(f"{parameter} successfully fetched")
    except Exception as e:
        logger.error(f"{e}")

def load_data(path:str)->pd.DataFrame:
    try:
        logger.debug(f"loading data from {path}")
        df = pd.read_csv(path,index_col=False)
        logger.debug(f"data loaded successfully")
        return df
    except Exception as e:
        logger.error(f"loading data unsuccessfull -> {e}")
        print(f"loading data unsuccessfull -> {e}")

def train_ordinal_encoder(data:pd.DataFrame)->None:
    try:
        logger.debug('training ordinal encoder')
        oe = OrdinalEncoder()
        oe.fit(data)
        joblib.dump(oe,'models/encoder.pkl')
        logger.debug('encoder saved as model/encoder.pkl')
    except Exception as e:
        logger.error(f"{e}")

def apply_encoder(data:pd.DataFrame)->pd.DataFrame:
    try:
        logger.debug('applying encoder')
        encoder = joblib.load('models/encoder.pkl')
        data = encoder.transform(data)
        logger.debug('encoder applied')
        return data
    except Exception as e:
        logger.error(f"{e}")

def save_data(path:str,dataname:str,data:pd.DataFrame)->None:
    try:
       
        
        logger.debug(f"creating a directiory as {path}")
        os.makedirs(path,exist_ok=True)
        logger.debug('directory created sucessfully')
        logger.debug(f"saving data on {path}")
        data.to_csv(os.path.join(path,dataname),index=False)
        logger.debug(f"data save successfully on path {path}")
    except Exception as e:
        logger.error(f"{e}")

# main 

def main():
    train_path = 'data/interim/train_preprocessed.csv'
    test_path = 'data/interim/test_preprocessed.csv'
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    en_cols = ['PARENT1', 'MSTATUS', 'GENDER', 'EDUCATION', 'OCCUPATION', 'CAR_USE',
       'CAR_TYPE', 'RED_CAR', 'REVOKED', 'URBANICITY']
    
    train_ordinal_encoder(train_data[en_cols])
    train_data[en_cols] = apply_encoder(train_data[en_cols]).astype('int')
    test_data[en_cols] = apply_encoder(test_data[en_cols]).astype('int')

    save_path = os.path.join('data','processed')
    save_data(save_path,'train_featured.csv',train_data)
    save_data(save_path,'test_featured.csv',test_data)

if __name__=="__main__":
    main()