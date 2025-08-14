import os
import pandas as pd
import joblib

from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import load_data,read_yaml

import mlflow
import mlflow.sklearn

logger = get_logger(__name__)


class ModelTraining:
    def __init__(self,train_path,test_path,model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS


    def load_split_data(self):

        try:
            logger.info(f"Loading data from {self.train_path}")
            train_df = pd.read_csv(self.train_path)
            logger.info(f"Loading data from {self.test_path}")
            test_df = pd.read_csv(self.test_path)

            X_train = train_df.drop(columns=["booking_status"])
            Y_train = train_df['booking_status']

            X_test = test_df.drop(columns=["booking_status"])
            Y_test = test_df["booking_status"]

            logger.info(f"Data Splitted Succesfully for Model training")

            return X_train,Y_train,X_test,Y_test

        except Exception as e :
            logger.error(f"Failed in loading data from file path {self.train_path} and {self.test_path}.")
            raise CustomException(f"Failed while loading data from the file location. ")


    def train_lgbm(self,X_train,Y_train):

        try:
            logger.info(f"Intializing our Model.")
            lgbm_model = lgb.LGBMClassifier(random_state = self.random_search_params['random_state'])

            logger.info("Starting our HyperParameter Tuning")

            random_search = RandomizedSearchCV(
                estimator = lgbm_model,
                param_distributions = self.params_dist,
                n_iter = self.random_search_params["n_iter"],
                cv = self.random_search_params["cv"],
                n_jobs = self.random_search_params["n_jobs"],
                verbose = self.random_search_params["verbose"],
                random_state = self.random_search_params["random_state"],
                scoring = self.random_search_params["scoring"]
            )

            logger.info(f"Starting our HyperParmameter Tuning")

            random_search.fit(X_train,Y_train)

            logger.info(f"Hyperparameter tuning completed ")

            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_

            logger.info(f"Best parameters are : {best_params}")

            return  best_lgbm_model

        except Exception as e :
            logger.error(f"Failed to train the model {e}")
            raise CustomException(f"Failed while training the model")

    def evaluate_model(self,model,X_test,Y_test):

        try:
            logger.info(f"Evaluating our model")

            Y_pred = model.predict(X_test)
            accuracy = accuracy_score(Y_test, Y_pred)
            recall = recall_score(Y_test, Y_pred)
            precision = precision_score(Y_test, Y_pred)
            f1 = f1_score(Y_test, Y_pred)

            logger.info(f"Accuracy Score: {accuracy}")
            logger.info(f"Precision Score: {precision}")
            logger.info(f"Recall Score: {recall}")
            logger.info(f"F1 Score: {f1}")

            return {
                "accuracy":accuracy,
                "precision":precision,
                "recall":recall,
                "f1":f1
            }
        except Exception as e :
            logger.error(f"Error While evaluating model {e}")
            raise CustomException(f"Failed to evaluate model",e)


    def save_model(self,model):

        try:
            os.makedirs(os.path.dirname(self.model_output_path),exist_ok=True)
            logger.info(f"Saving the model")
            joblib.dump(model,self.model_output_path)
            logger.info(f"Model Saved to {self.model_output_path}")

        except Exception as e:
            logger.error(f"Error while Saving Model {e}")
            raise CustomException(f"Failed to save model",e)


    def run(self):
        try:
            with mlflow.start_run():
                logger.info(f"Starting our model training Pipeline")
                logger.info(f"Starting our MLFLOW experimentation")

                logger.info(f"Logging the Training and Testing Dataset to MLFlow")
                mlflow.log_artifact(self.train_path,artifact_path ="datasets")
                mlflow.log_artifact(self.test_path, artifact_path="datasets")



                X_train, Y_train, X_test, Y_test = self.load_split_data()
                best_lgbm_model = self.train_lgbm(X_train,Y_train)
                metrics = self.evaluate_model(best_lgbm_model,X_test,Y_test)
                self.save_model(best_lgbm_model)
                logger.info(f"Logging the model into MLFLOW")

                mlflow.log_artifact(self.model_output_path)

                logger.info(f"Logging Params and Metrics into MLFLOW")
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)


                logger.info(f"Model Training Succesfully Completed ")

        except Exception as e:

            logger.error(f"Error in model training pipeline {e}")
            raise CustomException(f"Failed in pipeline ",e)


if __name__ =="__main__":

   trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH,MODEL_OUTPUT_PATH)
   trainer.run()