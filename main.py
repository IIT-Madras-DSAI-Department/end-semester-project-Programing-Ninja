import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Tuple

from algorithms import (
    MyStandardScaler,
    MyMultinomialLogisticRegression,
    MyTrueMulticlassSVM,
    MyKNeighborsClassifier,
    MyEnsembleClassifier
)

def load_all_data(base_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    train_orig_file = base_path + "MNIST_train.csv"
    validation_orig_file = base_path + "MNIST_validation.csv"
    test_orig_file = base_path + "MNIST_validation.csv"
    
    df_train_orig = pd.read_csv(train_orig_file)
    Xtrain, ytrain = df_train_orig.drop(columns=["label"]).values, df_train_orig["label"].values.astype(int)

    df_validation_orig = pd.read_csv(validation_orig_file)
    Xval, yval = df_validation_orig.drop(columns=["label"]).values, df_validation_orig["label"].values.astype(int)

    df_test_orig = pd.read_csv(test_orig_file)
    
    if "label" in df_test_orig.columns:
        Xtest = df_test_orig.drop(columns=["label"]).values
    else:
        Xtest = df_test_orig.values    
    return (Xtrain, ytrain, Xval, yval, Xtest)
     
base_path = r"/home/harshvardhan/ml_lab_project/end-semester-project-Programing-Ninja/"
Xtrain_orig, ytrain_orig, Xval_orig, yval_orig, Xtest_orig = load_all_data(base_path)

scaler = MyStandardScaler()
scaler.fit(Xtrain_orig)

X_train_scaled = scaler.transform(Xtrain_orig)
X_val_scaled = scaler.transform(Xval_orig)
X_test_scaled = scaler.transform(Xtest_orig)

eval_set_scaled = (X_val_scaled, yval_orig)


estimators = [
    MyMultinomialLogisticRegression(
        learning_rate=0.1, lambda_p=0.01, n_iters=1500, patience=10, validation_freq=10
    ),
    MyTrueMulticlassSVM(
        learning_rate=0.01, lambda_p=0.0001, n_iters=1500, patience=10, validation_freq=10
    ),
    MyKNeighborsClassifier(k=5)
]

ensemble_model = MyEnsembleClassifier(estimators=estimators)
ensemble_model.fit(X_train_scaled, ytrain_orig, eval_set=eval_set_scaled)

ypred_validation = ensemble_model.predict(X_val_scaled)

acc = accuracy_score(yval_orig, ypred_validation)
    
print(f"Ensemble Accuracy (on validation set): {acc:.4f}")


predict = ensemble_model.predict(X_test_scaled)
np.savetxt("test_predictions2.csv", predict, delimiter=",", fmt="%d", header="label", comments="")