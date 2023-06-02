
import os
import numpy as np
import pandas as pd
import tempfile

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from pickle import dump


# This is the location where the SageMaker Processing job
# will save the input dataset.
BASE_DIRECTORY = "/opt/ml/processing"
DATA_FILEPATH = Path(BASE_DIRECTORY) / "input"
TRAIN_DATA_FILEPATH =DATA_FILEPATH / "train" / "mnist_train.csv"
TEST_DATA_FILEPATH = DATA_FILEPATH / "test" / "mnist_test.csv"


def save_splits(base_directory, train, validation):
    """
    One of the goals of this script is to output the three
    dataset splits. This function will save each of these
    splits to disk.
    """
    
    train_path = Path(base_directory) / "train" 
    validation_path = Path(base_directory) / "validation"
    
    train_path.mkdir(parents=True, exist_ok=True)
    validation_path.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(train).to_csv(train_path / "train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(validation_path / "validation.csv", header=False, index=False)
    
    
def save_pipeline(base_directory, pipeline):
    """
    Saves the Scikit-Learn pipeline that we used to
    preprocess the data.
    """
    pipeline_path = Path(base_directory) / "pipeline"
    pipeline_path.mkdir(parents=True, exist_ok=True)
    dump(pipeline, open(pipeline_path / "pipeline.pkl", 'wb'))
    
def preprocess(base_directory, train_data_filepath, test_data_filepath):
    """
    Preprocesses the supplied raw dataset and splits it into a train, validation,
    and a test set.
    """
    
    df_train = pd.read_csv(train_data_filepath)
    df_test = pd.read_csv(test_data_filepath)
    
    X = df_train.copy()
    columns = list(X.columns)
    
    X = X.to_numpy()
    
    np.random.shuffle(X)
    
    train, validation = np.split(X, [int(.8 * len(X))])
    
    X_train = pd.DataFrame(train, columns=columns)
    X_validation = pd.DataFrame(validation, columns=columns)
    

    label_encoder = LabelEncoder()
    
    y_train = label_encoder.fit_transform(X_train.label)
    y_validation = label_encoder.transform(X_validation.label)
    
    X_train.drop(["label"], axis=1, inplace=True)
    X_validation.drop(["label"], axis=1, inplace=True)
    
    
    pipeline = Pipeline(steps=[
        ('preprocessor', ColumnTransformer(
            transformers=[('scaler', StandardScaler(), X_train.columns)]
        ))
    ])
    
    X_train = pipeline.fit_transform(X_train)
    X_validation = pipeline.transform(X_validation)
    
    train = np.concatenate((np.expand_dims(y_train, axis=1), X_train), axis=1)
    validation = np.concatenate((np.expand_dims(y_validation, axis=1), X_validation), axis=1)
    
    save_splits(base_directory, train, validation)

        

if __name__ == "__main__":
    preprocess(BASE_DIRECTORY, TRAIN_DATA_FILEPATH, TEST_DATA_FILEPATH)
