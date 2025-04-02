import pandas as pd 
import numpy as np 
import logging 
import os 
import joblib
import yaml
from sklearn.ensemble  import GradientBoostingClassifier


# configure logging
logger = logging.getLogger('model_building_log')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('logging.log')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_params():
    logger.debug('loading parameters file')
    try:
        with open('params.yaml','r') as file:
            file = yaml.safe_load(file)
            parameter = file['Model_building']
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

def model_training(data:pd.DataFrame,target:str,parameters:dict)->None:
    try:
        logger.debug('Creating GradientBoostingClassifier object')
        gbc = GradientBoostingClassifier(n_estimators=parameters['n_estimator'],max_depth=parameters['max_depth'],learning_rate=parameters['learning_rate'])
        logger.debug('Training the Model')
        gbc.fit(data.drop(columns=[target]),data[target])
        logger.debug('Saving the model as models/model.pkl')
        joblib.dump(gbc,'models/model.pkl')
        logger.debug('Model Trained Successfully')
    except Exception as e:
        logger.error(f"{e}")

# main 
def main():
    # load parameters

    param_dict = load_params()
    
    # load data
    data_path = 'data/processed/train_featured.csv'
    train_data = load_data(data_path)

    # train model 
    model_training(train_data,'claim',param_dict)

    
    
if __name__=='__main__':
    main()


# def model_loading()->GradientBoostingClassifier:
#     try:
#         logger.debug('Model Loading')
#         model = joblib.load('models/model.pkl')
#         logger.debug('Model Loaded Successfully')
#         return model
#     except Exception as e:
#         logger.error(f"{e}")

# def check_score(model:GradientBoostingClassifier,target,data:pd.DataFrame)->tuple[float,float,float]:
#     try:
#         logger.debug('Predicting....')
#         y_pred = model.predict(data.drop(columns=[target]))
#         y_pred_proba = model.predict_proba(data.drop(columns=[target]))
#         logger.debug('Prediction completed')
#         logger.debug('Measuring the Scores.....')
#         accuracy = accuracy_score(data[target],y_pred)
#         precision = precision_score(data[target],y_pred)
#         recall = recall_score(data[target],y_pred)
#         auc_score = roc_auc_score(data[target],y_pred[:,-1])
#         logger.debug('Score Measured')
#         scores = {
#             'accuracy':accuracy,
#             'precision':precision,
#             'recall':recall,
#             'auc_score':auc_score
#         }
#         joblib.dump(scores,'reports/metrics.json')
#         logger.debug('Score Stored on reports/metrics.json')
#         return accuracy,precision,recall,auc_score
#     except Exception as e:
#         logger.debug(f"{e}")




