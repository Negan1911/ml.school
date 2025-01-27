
import os
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

def train(base_directory, train_path, validation_path, epochs=50, batch_size=32, learning_rate=0.01):
    X_train = pd.read_csv(Path(train_path) / "train.csv")
    y_train = X_train[X_train.columns[0]]
    X_train.drop(X_train.columns[0], axis=1, inplace=True)
    
    X_validation = pd.read_csv(Path(validation_path) / "validation.csv")
    y_validation = X_validation[X_validation.columns[0]]
    X_validation.drop(X_validation.columns[0], axis=1, inplace=True)
    
    model = Sequential([
        Dense(128, input_shape=(X_train.shape[1],), activation="relu"),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=SGD(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train, 
        y_train, 
        validation_data=(X_validation, y_validation),
        epochs=epochs, 
        batch_size=batch_size,
        verbose=2,
    )

    predictions = np.argmax(model.predict(X_validation), axis=-1)
    print(f"Validation accuracy: {accuracy_score(y_validation, predictions)}")
    
    model_filepath = Path(base_directory) / "model" / "001"
    model.save(model_filepath)
    
if __name__ == "__main__":
    # Any hyperparameters provided by the training job are passed to the entry point
    # as script arguments. SageMaker will also provide a list of special parameters
    # that you can capture here. Here is the full list: 
    # https://github.com/aws/sagemaker-training-toolkit/blob/master/src/sagemaker_training/params.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_directory", type=str, default="/opt/ml/")
    parser.add_argument("--train_path", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", None))
    parser.add_argument("--validation_path", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", None))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    args, _ = parser.parse_known_args()
    
    train(
        base_directory=args.base_directory,
        train_path=args.train_path,
        validation_path=args.validation_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
