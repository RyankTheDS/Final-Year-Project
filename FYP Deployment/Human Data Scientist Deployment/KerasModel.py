# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 09:03:42 2024

@author: ryank
"""

from scikeras.wrappers import KerasClassifier
from tensorflow.keras import models, layers, regularizers, optimizers
from tensorflow.keras.constraints import MaxNorm
from keras.callbacks import EarlyStopping
from keras.metrics import AUC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
import pandas as pd
from keras.models import load_model

early_stopping = EarlyStopping(
    monitor='auc_2',   # Monitor validation auc
    min_delta=0.01,              # Minimum change to qualify as an improvement
    patience=5,              # Number of epochs to wait for improvement
    verbose=1,               # Print message when stopping is triggered
    mode='max',              # Mode for the metric (maximizing accuracy)
    baseline=None,           # No baseline in this example
    restore_best_weights=True # Restore best weights when stopping
)


# Function to train the model and return the fitted model
def train_model(X_train, y_train):
    model = configure_model()
    model.fit(X_train, y_train)
    return model

def test_keras_classifier(best_model, X_test, y_test):       
    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    print(f"Testing Keras")
    print(f"Test accuracy: {accuracy}")
    print(f"Test precision: {precision}")
    print(f"Test recall: {recall}")
    print(f"Test F1-score: {f1}")
    print(f"Test ROC-AUC score: {roc_auc}")
    print()

def configure_model():
    keras2 = load_model('Keras_Best_Model')
    scikeras2 = KerasClassifier(
        model = keras2,
        batch_size = 32,
        epochs = 50,
        loss="binary_crossentropy",
        optimizer="SGD",
        optimizer__learning_rate = 0.01,
        optimizer__momentum = 0.9,
        metrics=[AUC()], 
        callbacks=[early_stopping]
    )
    X_train_resampled = pd.read_csv("HumanRelated\X_train_resampled.csv")
    y_train_resampled = pd.read_csv("HumanRelated\y_train_resampled.csv")
    
    scikeras2.initialize(X_train_resampled, y_train_resampled)
    
    X_test = pd.read_csv("HumanRelated\X_test.csv")
    y_test = pd.read_csv("HumanRelated\y_test.csv")
    #model = train_model(X_train_resampled, y_train_resampled)
    test_keras_classifier(scikeras2, X_test, y_test)
    print("Test model created successfully.")
    return scikeras2

