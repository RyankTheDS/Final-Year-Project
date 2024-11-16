# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 14:15:13 2024

@author: ryank
"""

import streamlit as st
import requests
import json

# Custom CSS for styling the interface
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background-color: #f4f4f4;
    }

    /* Headers */
    .stHeader, .stTitle, .stSubheader, .stCaption, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #003366;
    }

    /* Main content background */
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #003366;
    }
    
    .css-1d391kg .stButton button {
        background-color: #0073e6;
        color: #ffffff;
    }
    
    /* Input fields */
    .stTextInput, .stNumberInput, .stSelectbox, .stSlider, .stRadio {
        background-color: #e6f2ff;
        color: #003366;
    }

    /* Buttons */
    .stButton button {
        background-color: #0073e6;
        color: #ffffff;
        border-radius: 5px;
        padding: 10px;
    }
    
    .stButton button:hover {
        background-color: #005bb5;
        color: #ffffff;
    }
    
    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 12px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1; 
    }

    ::-webkit-scrollbar-thumb {
        background: #888; 
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #555; 
    }
    
    /* Expander background and width */
    .streamlit-expanderHeader {
        background-color: #ffffff !important;
        padding: 10px !important;
        border-radius: 10px !important;
        margin-bottom: 10px !important;
    }

    .streamlit-expanderContent {
        background-color: #e6f2ff !important;
        padding: 20px !important;
        border-radius: 10px !important;
        max-width: 95% !important; /* Increase the width of the content */
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("Loan Default Prediction")

# Sidebar for important actions
st.sidebar.title("User Actions")
st.sidebar.write("Fill in the loan application details on the main screen and click 'Submit' to get the prediction.")

# Define mappings for categorical variables
status_checking_mapping = {
    "A11: ... < 0 DM": 0, 
    "A12: 0 <= ... < 200 DM": 1, 
    "A13: ... >= 200 DM / salary assignments for at least 1 year": 2, 
    "A14: no checking account": 3
}

credit_history_mapping = {
    "A30: no credits taken/all credits paid back duly": 0, 
    "A31: all credits at this bank paid back duly": 1, 
    "A32: existing credits paid back duly till now": 2, 
    "A33: delay in paying off in the past": 3, 
    "A34: critical account/other credits existing (not at this bank)": 4
}

purpose_mapping = {
    "A40: car (new)": 0, 
    "A41: car (used)": 1, 
    "A42: furniture/equipment": 3, 
    "A43: radio/television": 4, 
    "A44: domestic appliances": 5, 
    "A45: repairs": 6, 
    "A46: education": 7, 
    "A48: retraining": 8, 
    "A49: business": 9, 
    "A410: others": 2
}

savings_account_mapping = {
    "A61: ... < 100 DM": 0, 
    "A62: 100 <= ... < 500 DM": 1, 
    "A63: 500 <= ... < 1000 DM": 2, 
    "A64: ... >= 1000 DM": 3, 
    "A65: unknown/no savings account": 4
}

present_employment_mapping = {
    "A71: unemployed": 0, 
    "A72: ... < 1 year": 1, 
    "A73: 1 <= ... < 4 years": 2, 
    "A74: 4 <= ... < 7 years": 3, 
    "A75: ... >= 7 years": 4
}

personal_status_mapping = {
    "A91: male, divorced/separated": 0, 
    "A92: female, divorced/separated/married": 1, 
    "A93: male, single": 2, 
    "A94: male, married/widowed": 3, 
    "A95: female, single": 4
}

property_mapping = {
    "A121: real estate": 0, 
    "A122: if not A121: building society savings agreement/life insurance": 1, 
    "A123: if not A121/A122: car or other, not in attribute 6": 2, 
    "A124: unknown / no property": 3
}

# Input form for the deployment features
with st.form(key='loan_form'):
    st.header("Loan Application Details")
    
    # Grouping related inputs
    with st.expander("Personal Information"):
        status_checking = st.selectbox(
            "Status of existing checking account", 
            list(status_checking_mapping.keys()), 
            help="Current status of the applicant's checking account"
        )
        age = st.number_input(
            "Age in years", 
            min_value=18, max_value=100, 
            help="Age of the applicant"
        )
        personal_status = st.selectbox(
            "Personal status and sex", 
            list(personal_status_mapping.keys()), 
            help="Applicant's personal status and gender"
        )
    
    with st.expander("Loan Details"):
        credit_amount = st.number_input(
            "Credit amount", 
            min_value=0, max_value=1000000, 
            help="Total credit amount requested by the applicant"
        )
        duration = st.slider(
            "Duration in months", 
            min_value=6, max_value=72, 
            help="Duration of the loan in months"
        )
        purpose = st.selectbox(
            "Purpose of the loan", 
            list(purpose_mapping.keys()), 
            help="Purpose for which the loan is being requested"
        )
    
    with st.expander("Financial Information"):
        savings_account = st.selectbox(
            "Savings account/bonds", 
            list(savings_account_mapping.keys()), 
            help="Savings account or bonds held by the applicant"
        )
        present_employment = st.selectbox(
            "Present employment since", 
            list(present_employment_mapping.keys()), 
            help="Number of years the applicant has been in their current employment"
        )
    
    with st.expander("Credit History"):
        credit_history = st.selectbox(
            "Credit history", 
            list(credit_history_mapping.keys()), 
            help="Applicant's credit history"
        )
    
    with st.expander("Property Information"):
        property = st.selectbox(
            "Property", 
            list(property_mapping.keys()), 
            help="Property owned by the applicant"
        )
    
    # Submit button within the form
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    # Convert categorical inputs to numeric
    status_checking = status_checking_mapping[status_checking]
    credit_history = credit_history_mapping[credit_history]
    purpose = purpose_mapping[purpose]
    savings_account = savings_account_mapping[savings_account]
    present_employment = present_employment_mapping[present_employment]
    personal_status = personal_status_mapping[personal_status]
    property = property_mapping[property]
    
    # Send request to the API
    url1 = 'http://localhost:8000/predict'  # Update with actual endpoint
    data = {
        "Status_checking": status_checking,
        "Credit_amount": credit_amount,
        "Age": age,
        "Duration": duration,
        "Savings_account": savings_account,
        "Purpose": purpose,
        "Present_employment": present_employment,
        "Credit_history": credit_history,
        "Property": property,
        "Personal_status": personal_status
    }
    response = requests.post(url1, data=json.dumps(data))
    result = response.json()
    
    # Displaying the prediction result in a visually appealing way
    if result['prediction'] == 2:
        st.error("Prediction: Default")
    else:
        st.success("Prediction: No Default")
    
    # SHAP explanation (code unchanged as per your request)
    url2 = 'http://localhost:8000/shapplot'  # Update with actual endpoint
    response = requests.get(url2)
    shap_plot_html = response.text
    shap_html_with_background = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
        body {{ 
            background-color: white;
            font-family: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Helvetica Neue", Arial, sans-serif; 
        }}
        .scroll-container {{
            width: 100%;
            overflow-x: auto;
            overflow-y: hidden;
            white-space: nowrap;
        }}
        .shap-content {{
            display: inline-block;
            min-width: 2200px;  
        }}
        </style>
    </head>
    <body>
    <div class="scroll-container">
        <h4>SHAP Force Plot for the Entered Data</h4>
        <div class="shap-content">
        {shap_plot_html}
        </div>
    </div>
    </body>
    </html>
    """
    st.components.v1.html(shap_html_with_background, height=230, scrolling=True)
