# Import libraries

import argparse
import glob
import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
# import logging
import sys
# from autologging import logged, TRACE, traced
import mlflow
# from mlflow.models import infer_signature
# from mlflow.utils.environment import _mlflow_conda_env


# define functions.
# @traced
# @logged
def main(args):
    # TO DO: enable autologging
    logs = mlflow.autolog()    

    # read data
    df = get_csvs_df(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    model = train_model(args.reg_rate, data["train"]["X"], data["test"]["X"], data["train"]["y"], data["test"]["y"])
    
    get_model_metrics
    metrics = get_model_metrics(model, data)
    # main._log.info("This is an info for main_message")

# @traced
# @logged
def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)
    
    # get_csvs_df._log.info("This is an info for get_csvc_df_message")


# TO DO: add function to split data
# @traced
# @logged
def split_data(df):
    X = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values
    y = df['Diabetic'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    data = {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}}
    return data

    # split_data._log.info("This is an info for split_data_message")

# @traced
# @logged
# Train the model, return the model
def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # train model
    model = LogisticRegression(C=1/reg_rate, solver="liblinear").fit(data["train"]["X"], data["train"]["y"])
    return model
    # train_model._log.info("This is an info for train_model_message")

# Evaluate the metrics for the model
# @traced
# @logged
def get_model_metrics(reg_model, data):
    preds_prob = model.predict_proba(data["test"]["X"])
    preds_deci = model.decision_function(data["test"]["X"])
    auc_prob = roc_auc_score(preds_prob[:,1], data["test"]["y"])
    auc_deci = roc_auc_score(preds_deci, data["test"]["y"])
    metrics = {"auc_prob": auc_prob,
               "auc_deci": auc_deci}
    return metrics
    # get_model_metrics._log.info("This is an info for get_model_metrics_message")

# @traced
# @logged
def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args
    
    # parse_args._log.info("This is an info for parse_args_message")

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
