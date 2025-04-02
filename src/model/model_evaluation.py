import pandas as pd 
import numpy as np 
import os 
import joblib
import logging
import json

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score , recall_score, precision_score , roc_auc_score

# configure logging
logger = logging.getLogger('model_evaluation_log')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('logging.log')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)




def load_data(path:str)->pd.DataFrame:
    try:
        logger.debug(f"loading data from {path}")
        df = pd.read_csv(path,index_col=False)
        logger.debug(f"data loaded successfully")
        return df
    except Exception as e:
        logger.error(f"loading data unsuccessfull -> {e}")
        print(f"loading data unsuccessfull -> {e}")


def model_loading()->GradientBoostingClassifier:
    try:
        logger.debug('Model Loading')
        model = joblib.load('models/model.pkl')
        logger.debug('Model Loaded Successfully')
        return model
    except Exception as e:
        logger.error(f"{e}")

def check_score(model:GradientBoostingClassifier,target,data:pd.DataFrame)->tuple[float,float,float]:
    try:
        logger.debug('Predicting....')
        y_pred = model.predict(data.drop(columns=[target]))
        y_pred_proba = model.predict_proba(data.drop(columns=[target]))
        logger.debug('Prediction completed')
        logger.debug('Measuring the Scores.....')
        accuracy = accuracy_score(data[target],y_pred)
        precision = precision_score(data[target],y_pred)
        recall = recall_score(data[target],y_pred)
        auc_score = roc_auc_score(data[target],y_pred_proba[:,-1])
        logger.debug('Score Measured')
        scores = {
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'auc_score':auc_score
        }
        with open('reports/metrics.json', 'w') as f:
            json.dump(scores, f, indent=4)  # `indent=4` makes it more readable
        logger.debug('Score Stored on reports/metrics.json')
        return scores
    except Exception as e:
        logger.debug(f"{e}")


# main

def main():
    data_path = 'data/processed/test_featured.csv'
    test_data = load_data(data_path)
    model = model_loading()
    scores = check_score(model,'claim',test_data)
    return scores


    

if __name__=='__main__':
    scores = main()
    print(scores)