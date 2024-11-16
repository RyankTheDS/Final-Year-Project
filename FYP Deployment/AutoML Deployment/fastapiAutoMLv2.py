# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 12:38:28 2024

@author: ryank
"""

from fastapi import FastAPI
from pydantic import BaseModel
import h2o
#from h2o.frame import H2OFrame
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import io
import shap
import matplotlib.pyplot as plt
from fastapi.responses import Response
from matplotlib.ticker import FuncFormatter

app = FastAPI()

# To initialize H2O
h2o.init()

# To load MOJO model 
model = h2o.import_mojo('Best_AutoML_model4.zip')



# To load the Standard Scaler object
with open('AutoMLRelated\standard_scaler.pkl', 'rb') as f:
    scaler_obj = pickle.load(f)



class LoanApplication(BaseModel):
    checking_acct_status: str
    credit_history: str
    purpose: str
    savings_acct_bonds: str
    employment_duration: str
    personal_status_sex: str
    other_debtors_guarantors: str
    other_installment_plans: str
    existing_credits_at_bank: int
    no_of_people_liable: int
    telephone: str
    foreign_worker: str
    duration_category: str
    age_category: str


# To define label encoding mappings
label_mappings = {
    'checking_acct_status': {'A11': 0, 'A12': 1, 'A13': 2, 'A14': 3},
    'credit_history': {'A30': 0, 'A31': 1, 'A32': 2, 'A33': 3, 'A34': 4},
    'purpose': {'A40': 0, 'A41': 1, 'A410': 2, 'A42': 3, 'A43': 4, 'A44': 5, 'A45': 6, 'A46': 7, 'A47': 8, 'A48': 9, 'A49': 10},
    'savings_acct_bonds': {'A61': 0, 'A62': 1, 'A63': 2, 'A64': 3, 'A65': 4},
    'employment_duration': {'A71': 0, 'A72': 1, 'A73': 2, 'A74': 3, 'A75': 4},
    'personal_status_sex': {'A91': 0, 'A92': 1, 'A93': 2, 'A94': 3, 'A95': 4},
    'other_debtors_guarantors': {'A101': 0, 'A102': 1, 'A103': 2},
    'other_installment_plans': {'A141': 0, 'A142': 1, 'A143': 2},
    'telephone': {'A191': 0, 'A192': 1},
    'foreign_worker': {'A201': 0, 'A202': 1},
    'duration_category': {'Short term': 1, 'Medium term': 2, 'Long term': 3},
    'age_category': {'20s and lower': 1, '30s': 2, '40s': 3, '50s': 4, 'Seniors': 5}
}



# Define a function to apply label encoding
def apply_label_encoding(column, mapping):
    column = column.replace(mapping)
    return column

#def shorten_y_labels(y, pos):
    #return f'{y:.2e}'

@app.post('/predict')
def predict_loan_default(application: LoanApplication):     
        
    # Convert application data to a DataFrame
    data = {
        'checking_acct_status': [application.checking_acct_status],
        'credit_history': [application.credit_history],
        'purpose': [application.purpose],
        'savings_acct_bonds': [application.savings_acct_bonds],
        'employment_duration': [application.employment_duration],
        'personal_status_sex': [application.personal_status_sex],
        'other_debtors_guarantors': [application.other_debtors_guarantors],
        'other_installment_plans': [application.other_installment_plans],
        'existing_credits_at_bank': [application.existing_credits_at_bank],
        'no_of_people_liable': [application.no_of_people_liable],
        'telephone': [application.telephone],
        'foreign_worker': [application.foreign_worker],
        'duration_category': [application.duration_category],
        'age_category': [application.age_category]
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame(data)
    print(input_df)
    # Create the ColumnTransformer
    preprocessing_steps = [
        ('checking_acct_status', FunctionTransformer(lambda x: apply_label_encoding(x, label_mappings['checking_acct_status']), validate=False), ['checking_acct_status']),
        ('credit_history', FunctionTransformer(lambda x: apply_label_encoding(x, label_mappings['credit_history']), validate=False), ['credit_history']),
        ('purpose', FunctionTransformer(lambda x: apply_label_encoding(x, label_mappings['purpose']), validate=False), ['purpose']),
        ('savings_acct_bonds', FunctionTransformer(lambda x: apply_label_encoding(x, label_mappings['savings_acct_bonds']), validate=False), ['savings_acct_bonds']),
        ('employment_duration', FunctionTransformer(lambda x: apply_label_encoding(x, label_mappings['employment_duration']), validate=False), ['employment_duration']),
        ('personal_status_sex', FunctionTransformer(lambda x: apply_label_encoding(x, label_mappings['personal_status_sex']), validate=False), ['personal_status_sex']),
        ('other_debtors_guarantors', FunctionTransformer(lambda x: apply_label_encoding(x, label_mappings['other_debtors_guarantors']), validate=False), ['other_debtors_guarantors']),
        ('other_installment_plans', FunctionTransformer(lambda x: apply_label_encoding(x, label_mappings['other_installment_plans']), validate=False), ['other_installment_plans']),
        ('existing_credits_at_bank', FunctionTransformer(lambda x: x, validate=False), ['existing_credits_at_bank']),
        ('no_of_people_liable', FunctionTransformer(lambda x: x, validate=False), ['no_of_people_liable']),
        ('telephone', FunctionTransformer(lambda x: apply_label_encoding(x, label_mappings['telephone']), validate=False), ['telephone']),
        ('foreign_worker', FunctionTransformer(lambda x: apply_label_encoding(x, label_mappings['foreign_worker']), validate=False), ['foreign_worker']),
        ('duration_category', FunctionTransformer(lambda x: apply_label_encoding(x, label_mappings['duration_category']), validate=False), ['duration_category']),
        ('age_category', FunctionTransformer(lambda x: apply_label_encoding(x, label_mappings['age_category']), validate=False), ['age_category'])
    ]
    preprocessor = ColumnTransformer(transformers=preprocessing_steps)
    # Preprocess the data 
    preprocessed_data = preprocessor.fit_transform(input_df)
    print(len(preprocessed_data))
    # Convert preprocessed data to DataFrame
    preprocessed_df = pd.DataFrame(preprocessed_data, columns=['checking_acct_status', 'credit_history', 'purpose', 'savings_acct_bonds', 'employment_duration', 'personal_status_sex', 'other_debtors_guarantors', 'other_installment_plans', 'existing_credits_at_bank', 'no_of_people_liable', 'telephone', 'foreign_worker', 'duration_category', 'age_category'])
    print(preprocessed_df)
    for column in preprocessed_df.columns:
        print(f"Values in column '{column}':")
        for value in preprocessed_df[column]:
            print(value)
        print()
    print(preprocessed_df.info())
    scaled_df = scaler_obj.transform(preprocessed_df)
    scaled_df = pd.DataFrame(scaled_df, columns=preprocessed_df.columns, index=preprocessed_df.index)
    print(scaled_df.info())
    values = scaled_df.values.flatten().tolist()
    single_row = h2o.H2OFrame({
        'checking_acct_status': [None],
        'credit_history': [None],
        'purpose': [None],
        'savings_acct_bonds': [None],
        'employment_duration': [None],
        'personal_status_sex': [None],
        'other_debtors_guarantors': [None],
        'other_installment_plans': [None],
        'existing_credits_at_bank': [None],
        'no_of_people_liable': [None],
        'telephone': [None],
        'foreign_worker': [None],
        'duration_category': [None],
        'age_category': [None]
    })
    for i, column in enumerate(single_row.columns):
        single_row[column] = values[i]
    # Convert preprocessed data to H2OFrame
    #h2o_df = h2o.H2OFrame(preprocessed_df)
    
    # Make prediction
    prediction = model.predict(single_row)
    print(prediction)
    # Extract predicted values
    pred_values = prediction.as_data_frame().values.tolist()
    global user_input
    user_input = single_row
    #single_row["credit_risk"] = prediction["predict"]
    #shap_values = model.explain_row(single_row, include_explanations=["shap_explain_row"], row_index=0)
    #img_bytes = io.BytesIO()
    #shap.summary_plot(shap_values, show=False).save(img_bytes, format='PNG')
    #img_bytes.seek(0)
    return {
        "predictions": pred_values
    }

@app.post("/show-shap")
async def show_shap():
    plt.figure()
    model.shap_explain_row_plot(user_input, row_index=0)
    plt.title("Explanation for Prediction", fontsize=16)
    plt.xlabel("SHAP Value", fontsize=14)
    plt.ylabel("Features", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # Save plot to a BytesIO object
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()    
    
    return Response(content=buf.getvalue(), media_type="image/png")
    
    
    
    
    