# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 12:21:57 2024

@author: ryank
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
from fastapi.responses import FileResponse

app = FastAPI()

# Load the model
model = joblib.load('GenAIRelated/final_best_model.pkl')

# Load and adjust the scaler
scaler = joblib.load('GenAIRelated/final_scaler.pkl')
X_train = pd.read_csv("GenAIRelated/fin_gen_X_train_resampled.csv")

# Define the selected features
selected_features = ['Status_checking', 'Credit_amount', 'Age', 'Duration', 'Savings_account',
                     'Purpose', 'Present_employment', 'Credit_history', 'Property', 'Personal_status']

class Application(BaseModel):
    Status_checking: int
    Credit_amount: float
    Age: int
    Duration: int
    Savings_account: int
    Purpose: int
    Present_employment: int
    Credit_history: int
    Property: int
    Personal_status: int

@app.get("/shapplot")
def show_shap():
    # SHAP explanation
    explainer = shap.TreeExplainer(model, X_train[selected_features])
    shap_values = explainer.shap_values(data_selected)
    shap.save_html('gen_shap_force_plot.html', shap.force_plot(explainer.expected_value, shap_values[0], data_selected))
    return FileResponse("gen_shap_force_plot.html", media_type='text/html')

@app.post("/predict")
async def predict(application: Application):
    # Convert application data to DataFrame with selected features only
    data = pd.DataFrame([[
        application.Status_checking, application.Credit_amount, application.Age, application.Duration,
        application.Savings_account, application.Purpose, application.Present_employment,
        application.Credit_history, application.Property, application.Personal_status
    ]], columns=selected_features)
    # data = {
    # 'Status_checking': [0], 
    # 'Credit_amount': [0], 
    # 'Age': [19], 
    # 'Duration': [6], 
    # 'Savings_account': [0], 
    # 'Purpose': [0], 
    # 'Present_employment': [0], 
    # 'Credit_history': [0], 
    # 'Property': [0], 
    # 'Personal_status': [0]
    # }    
    # print(data)
    # data = pd.DataFrame(data)
    print(data)
    # Scale the data
    print(scaler)
    data_scaled = scaler.transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=selected_features)
    print(data_scaled)
    # Keep only selected features (though all columns are selected)
    global data_selected
    data_selected = data_scaled
    
    # Predict using the model
    prediction = model.predict(data_selected)
    print(prediction)
    return {"prediction": int(prediction[0])}
