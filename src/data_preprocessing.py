import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from  config.paths_config import *
from utils.common_functions import read_yaml,load_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


logger = get_logger(__name__)


class DataProcessor:

    def __init__(self,train_path,test_path,processed_dir,config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir =processed_dir

        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)


    def preprocess_data(self,df):
        try:
            logger.info("Starting the Data Processing Step")
            logger.info("Dropping the columns ")
            df.drop(columns=['Unnamed: 0', 'Booking_ID'], axis=1, inplace=True)
            df.drop_duplicates(inplace=True)

            cat_cols = self.config['data_processing']['categorical_columns']
            num_cols = self.config['data_processing']['numerical_columns']

            logger.info("Applying Label Encoding")

            label_encoder = LabelEncoder()
            mappings = {}

            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])
                mappings[col] = {label: code for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}

            logger.info(f"Label Mapping are : {mappings}")

            logger.info("Doing Skewness Handling")

            skew_threshold = self.config['data_processing']['skewness_threshold']
            skewness = df[num_cols].apply(lambda x :x.skew())

            for column in skewness[skewness>skew_threshold].index:
                df[column] = np.log1p(df[column])

            return df

        except Exception as e :
            logger.error(f"Error during preprocess step {e}")
            raise CustomException(f"Error in Preprocess Data.")


    def balance_data(self,df):
        try:
            logger.info("Handling Imbalanced Data")

            X = df.drop('booking_status', axis=1)
            y = df['booking_status']

            smote = SMOTE(random_state =42)
            X_resample,Y_resample = smote.fit_resample(X,y)

            balanced_df = pd.concat([X_resample,Y_resample], axis=1)

            logger.info(f"Data balanced Succesfully")

            return balanced_df
        except Exception as e :
            logger.error(f"Error During Balancing Data step {e}")
            raise  CustomException (f'Error During Balance Data.')

    def select_features(self,df):
        try:
            logger.info(f"Starting our Feature Selection Step.")
            X = df.drop('booking_status', axis=1)
            y = df['booking_status']

            model = RandomForestClassifier(random_state =42)
            model.fit(X,y)

            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': X.columns,
                                                  'Importance': feature_importance})

            top_features_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

            num_features_to_select = self.config['data_processing']['no_of_features']
            top_10_feature = top_features_importance_df['Feature'].head(num_features_to_select).values

            logger.info(f"Top 10 Features :{top_10_feature}")

            top_10_df = df[top_10_feature.tolist() + ['booking_status']]

            logger.info(f"Feature selection completed Successfully")

            return  top_10_df
        except Exception as e:
            logger.error(f"Error during Feature Selection {e}")
            raise CustomException("Error during Feature Selection",e)


    def save_data(self,df,file_path):

        try:
            logger.info("Saving our data in Processed Folder")

            df.to_csv(file_path,index=False)

            logger.info(f"Data Saved Successfully to {file_path}")

        except Exception as e:
            logger.error(f"Error during Saving data step {e}")
            raise  CustomException(f"Error while Saving Data ",e)


    def process(self):

        try:
            logger.info(f"Loading data from RAW Directory")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)

            train_df = self.select_features(train_df)
            test_df = test_df[train_df.columns]

            self.save_data(train_df,PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df,PROCESSED_TEST_DATA_PATH)

            logger.info(f"Data Processing Completed Successfully")

        except Exception as e:
            logger.error(f"Error while preprocessing pipeline {e}")
            raise  CustomException(f"Error while data Preprocessing pipeline",e)




if __name__ == "__main__":

    processor = DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,PROCESSED_DIR,CONFIG_PATH)
    processor.process()









