import pandas as pd 
import numpy as np 
import logging
import os 

from sklearn.preprocessing import StandardScaler
import  joblib


# configure logging
logger = logging.getLogger('Data_preprocessing_log')
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
            parameter = file['Data_ingestion'][parameter] # there should be data_preprocessing params at this time i leave it .
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

def dropnullvalues(data:pd.DataFrame)->pd.DataFrame:
    try:
        logger.debug('Dropping null values')
        data = data.dropna()
        return data
        logger.debug('Null values dropped')
    except Exception as e:
        logger.error(f"{e}")

def train_standard_scaler(data:pd.DataFrame)->pd.DataFrame:
    try:
        logging.debug('Creating StandardScalerobject')
        st = StandardScaler()
        logging.debug('training StandardScalerobject')
        st.fit(data)
        logging.debug('Saving StandardScaler object ')
        joblib.dump(st,'models/scaler.pkl')
        logging.debug('Ss dump on models/scaler.pkl')
    except Exception as e:
        logging.error(f"{e}")

def applyscaler(data:pd.DataFrame)->pd.DataFrame:
    try:
        logger.debug('loading model')
        # load scaler
        model = joblib.load('models/scaler.pkl')
        # apply
        logger.debug('applying scaler')
        data = model.transform(data)
        return data
        logger.debug('scaling applied successfully')
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
    train_path = 'data/raw/train.csv'
    test_path = 'data/raw/test.csv'
    train_data = load_data(train_path)
    print(train_data.head())
    test_data = load_data(test_path)
    train_data = dropnullvalues(train_data)
    test_data = dropnullvalues(test_data)
    scale_cols = ['AGE','YOJ','INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CAR_AGE']
    train_standard_scaler(train_data[scale_cols])
    train_data[scale_cols] = applyscaler(train_data[scale_cols])
    test_data[scale_cols] = applyscaler(test_data[scale_cols])

    save_path = os.path.join('data','interim')

    save_data(save_path,'train_preprocessed.csv',train_data)
    save_data(save_path,'test_preprocessed.csv',test_data)
    
    
if __name__=='__main__':
    main()