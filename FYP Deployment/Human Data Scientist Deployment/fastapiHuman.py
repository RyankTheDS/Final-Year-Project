# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 20:22:25 2024

@author: ryank
"""

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import io
import shap
from tensorflow.keras import models, layers, regularizers, optimizers
from tensorflow.keras.constraints import MaxNorm
from scikeras.wrappers import KerasClassifier
from KerasModel import configure_model
from fastapi.responses import Response
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt

#from kerascode import create_keras_model
#
app = FastAPI()


best_model = configure_model()

# To load the Standard Scaler object
with open('HumanRelated\human_standard_scaler.pkl', 'rb') as f:
    scaler_obj = pickle.load(f)

class LoanApplication(BaseModel):
    checking_acct_status: str
    credit_history: str
    savings_acct_bonds: str
    employment_duration: str
    personal_status_sex: str
    other_installment_plans: str
    housing: str
    telephone: str
    duration_category: str
    purpose: str
    property: str
    present_residence_duration: int
    foreign_worker: str
    existing_credits_at_bank: int

# To define label encoding mappings
label_mappings = {
    'checking_acct_status': {'A11': 0, 'A12': 1, 'A13': 2, 'A14': 3},
    'credit_history': {'A30': 0, 'A31': 1, 'A32': 2, 'A33': 3, 'A34': 4},
    'savings_acct_bonds': {'A61': 0, 'A62': 1, 'A63': 2, 'A64': 3, 'A65': 4},
    'employment_duration': {'A71': 0, 'A72': 1, 'A73': 2, 'A74': 3, 'A75': 4},
    'personal_status_sex': {'A91': 0, 'A92': 1, 'A93': 2, 'A94': 3, 'A95': 4},
    'other_installment_plans': {'A141': 0, 'A142': 1, 'A143': 2},
    'housing': {'A151': 0, 'A152': 1, 'A153': 2},
    'telephone': {'A191': 0, 'A192': 1},
    'duration_category': {"Short term": 1, "Medium term": 2, "Long term": 3},
    'purpose': {'A40': 0, 'A41': 1, 'A410': 2, 'A42': 3, 'A43': 4, 'A44': 5, 'A45': 6, 'A46': 7, 'A47': 8, 'A48': 9, 'A49': 10},
    'property': {'A121': 0, 'A122': 1, 'A123': 2, 'A124': 3},
    'foreign_worker': {'A201': 0, 'A202': 1}
}
    
# Define a function to apply label encoding
def apply_label_encoding(column, mapping):
    column = column.replace(mapping)
    return column

@app.post('/predict')
def predict_loan_default(application: LoanApplication):
    data = {
        'checking_acct_status': [application.checking_acct_status],
        'credit_history': [application.credit_history],
        'savings_acct_bonds': [application.savings_acct_bonds],
        'employment_duration': [application.employment_duration],
        'personal_status_sex': [application.personal_status_sex],
        'other_installment_plans': [application.other_installment_plans],
        'housing': [application.housing],
        'telephone': [application.telephone],
        'duration_category': [application.duration_category],
        'purpose': [application.purpose],
        'property': [application.property],
        'present_residence_duration': [application.present_residence_duration],
        'foreign_worker': [application.foreign_worker],
        'existing_credits_at_bank': [application.existing_credits_at_bank]
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame(data)
    print(input_df)
    # Create the ColumnTransformer
    preprocessing_steps = [
        ('checking_acct_status', FunctionTransformer(lambda x: apply_label_encoding(x, label_mappings['checking_acct_status']), validate=False), ['checking_acct_status']),
        ('credit_history', FunctionTransformer(lambda x: apply_label_encoding(x, label_mappings['checking_acct_status']), validate=False), ['checking_acct_status']),
        ('savings_acct_bonds', FunctionTransformer(lambda x: apply_label_encoding(x, label_mappings['savings_acct_bonds']), validate=False), ['savings_acct_bonds']),
        ('employment_duration', FunctionTransformer(lambda x: apply_label_encoding(x, label_mappings['employment_duration']), validate=False), ['employment_duration']),
        ('personal_status_sex', FunctionTransformer(lambda x: apply_label_encoding(x, label_mappings['personal_status_sex']), validate=False), ['personal_status_sex']),
        ('other_installment_plans', FunctionTransformer(lambda x: apply_label_encoding(x, label_mappings['other_installment_plans']), validate=False), ['other_installment_plans']),
        ('housing', FunctionTransformer(lambda x: apply_label_encoding(x, label_mappings['housing']), validate=False), ['housing']),
        ('telephone', FunctionTransformer(lambda x: apply_label_encoding(x, label_mappings['telephone']), validate=False), ['telephone']),
        ('duration_category', FunctionTransformer(lambda x: apply_label_encoding(x, label_mappings['duration_category']), validate=False), ['duration_category']),
        ('purpose', FunctionTransformer(lambda x: apply_label_encoding(x, label_mappings['purpose']), validate=False), ['purpose']),
        ('property', FunctionTransformer(lambda x: apply_label_encoding(x, label_mappings['property']), validate=False), ['property']),
        ('present_residence_duration', FunctionTransformer(lambda x: x, validate=False), ['present_residence_duration']),
        ('foreign_worker', FunctionTransformer(lambda x: apply_label_encoding(x, label_mappings['foreign_worker']), validate=False), ['foreign_worker']),
        ('existing_credits_at_bank', FunctionTransformer(lambda x: x, validate=False), ['existing_credits_at_bank'])
    ]
    preprocessor = ColumnTransformer(transformers=preprocessing_steps)
    # Preprocess the data 
    preprocessed_data = preprocessor.fit_transform(input_df)
    print(len(preprocessed_data))
    # Convert preprocessed data to DataFrame
    preprocessed_df = pd.DataFrame(preprocessed_data, columns=['checking_acct_status', 'credit_history', 'savings_acct_bonds', 'employment_duration', 'personal_status_sex', 'other_installment_plans', 'housing', 'telephone', 'duration_category', 'purpose', 'property', 'present_residence_duration', 'foreign_worker', 'existing_credits_at_bank'])
    print(preprocessed_df)
    for column in preprocessed_df.columns:
        print(f"Values in column '{column}':")
        for value in preprocessed_df[column]:
            print(value)
        print()
    print(preprocessed_df.info())
    global scaled_df
    scaled_df = scaler_obj.transform(preprocessed_df)
    scaled_df = pd.DataFrame(scaled_df, columns=preprocessed_df.columns, index=preprocessed_df.index)
    print(scaled_df.info())
    global prediction
    prediction = best_model.predict(scaled_df)
    print(prediction.tolist())
    return {
        'prediction': prediction.tolist()
    }

@app.get("/show-shap")
def show_shap():
    X_train_resampled = pd.read_csv("HumanRelated/X_train_resampled.csv")
    # Initialize SHAP explainer
    explainer = shap.KernelExplainer(best_model.predict, X_train_resampled)  # Use `predict_proba` for classification
    shap_values = explainer.shap_values(scaled_df)

    # Save the SHAP force plot as an HTML file
    shap.save_html('shap_force_plot.html', shap.force_plot(explainer.expected_value, shap_values[0], scaled_df))
    return FileResponse("shap_force_plot.html", media_type='text/html')
    
# @app.post("/show-shap")
# async def show_shap():
#      X_train_resampled = pd.read_csv("HumanRelated/X_train_resampled.csv")
#      #X_train_resampled = pd.concat([X_train_resampled, scaled_df.iloc[0]], ignore_index=True)
#      #explainer = shap.KernelExplainer(f, X_train_resampled.iloc[:50,:])
#      shap.initjs()
#      explainer = shap.KernelExplainer(best_model.predict, X_train_resampled.iloc[:50,:])
#      #shap_values = explainer.shap_values(X_train_resampled.iloc[[-1]], nsamples=500)
#      shap_values = explainer.shap_values(scaled_df)
#      plt.figure()
#      shap.force_plot(explainer.expected_value, shap_values[1], scaled_df)
#      plt.savefig('shap_force_plot.png', format='png')
#      buf = io.BytesIO()    
#      plt.savefig(buf, format='png', bbox_inches='tight')
#      buf.seek(0)
#      plt.close()    
# #     print(100)
#      return Response(content=buf.getvalue(), media_type="image/png")
    #return StreamingResponse(buf, media_type='image/png')
    
    
    
    
    
    
    
    
    
    
    
    
    
