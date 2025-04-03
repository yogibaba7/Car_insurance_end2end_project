import pandas as pd 
import numpy as np 
import os 
import joblib
import logging
import json
from dvclive import Live
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score , recall_score, precision_score , roc_auc_score


# configure dagshub
import mlflow
import dagshub
dagshub.init(repo_owner='yogibaba7', repo_name='Car_insurance_end2end_project', mlflow=True)
# creating experiment
mlflow.set_experiment('experiment1')


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

        # log metrics
        
        mlflow.log_metrics(scores)
        mlflow.log_artifact(__file__)
        return scores
    except Exception as e:
        logger.debug(f"{e}")


# main
def main():
    # get the run_id 
    try:
        with open('run_id.txt','r') as file:
            run_id = file.read().strip()
    except :
        run_id = None
    
    with mlflow.start_run(run_id=run_id , nested=False) as parent_run:

        data_path = 'data/processed/test_featured.csv'
        test_data = load_data(data_path)
        model = model_loading()
        scores = check_score(model,'claim',test_data)
        
        with Live(save_dvc_exp=True) as live:
            for name,value in scores.items():
                live.log_metric(name,value)  

        # add tags
        mlflow.set_tag('author','yogesh')
        mlflow.set_tag('version','v1.0')
        mlflow.set_tag('model','baseline')
                    

if __name__=='__main__':
    main()
        