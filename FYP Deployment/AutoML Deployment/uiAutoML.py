# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:06:40 2024

@author: ryank
"""

import streamlit as st
import requests
from PIL import Image
from io import BytesIO


# Set the title of the app
st.title('Loan Default Predictor')

def main():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://static.vecteezy.com/system/resources/previews/005/374/564/original/illustration-graphic-cartoon-character-of-money-saving-of-bank-free-vector.jpg");
            background-size: contain;
            background-position: 80% 20%; 
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .stApp h1 {
            color: black;
            text-align: center;
            background-color: rgba(255, 255, 255, 0.8);
        }
        [data-testid="stSidebar"] {
            background-color: white;
            border: 2px solid #333; /* Border color */
            border-radius: 10px; /* Curved edges */
        }
        
        </style>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown("## What would you like to do?")
    step = st.sidebar.radio("Choose an Option", ["About the App","Make Prediction"])
    st.sidebar.markdown(
        """
        <br/>
        <img src="https://miro.medium.com/v2/resize:fit:1200/1*BFtMT2-yL6MkiWII4ySuqw.jpeg" alt="Image" style="width: 100%;">
        """
        , unsafe_allow_html=True
    )
    if step == "About the App":
        introduce()
    else:
        get_info_page()
        

def introduce():
    # Custom CSS for the paragraph
    st.markdown(
        """
        <style>
        .info-box {
            background-color: #f0f0f0; /* Light gray background */
            padding: 20px;
            border-radius: 10px; /* Curved edges */
            color: black;
            font-size: 17px;
            width: 100%
        }
        .info-box ul li {
            font-size: 17px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display the paragraph
    st.markdown(
        """
        <div class="info-box">
            This app is designed to help users predict loan defaults. Key features include:
            <ul>
                <li><b>Advanced Machine Learning Models:</b> Leverages sophisticated algorithms to analyse borrower attributes and financial metrics.</li>
                <li><b>Accurate Assessment:</b> Provides a precise evaluation of default risk, aiding financial institutions in making informed decisions.</li>
                <li><b>User-Friendly Interface:</b> Easy-to-use interface ensures a smooth experience for all users.</li>
                <li><b>Valuable Tool:</b> Ideal for financial institutions, lenders and analysts to effectively manage and mitigate risks.</li>
                <li><b>Efficiency and Reliability:</b> Streamlines the process of evaluating loan applications, enhancing both speed and accuracy.</li>
            </ul>
                With the <b>Loan Default Predictor</b> app, users can expect a robust, reliable and efficient tool for assessing loan default risks.
            </div>
            """,
            unsafe_allow_html=True
    )

def check_response():
    for answer in st.session_state.answers:
        if answer is None:
            return False
    return True

def update_response(index):
    if st.session_state.answers[index] is not None:
        st.session_state.is_valid[index] = True
        st.success("✔️ Answered")
    else:
        st.session_state.is_valid[index] = False
        st.error("Please enter a value.", icon="🚨")

#def reset_form():
    #st.session_state.is_valid = [True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    #st.session_state.answers = [None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    #st.session_state.prediction = None
    
def get_prediction(data):
    url = 'http://localhost:8000/predict'
    response = requests.post(url, json=data)
    #if response.any():
        #st.write("OK")
    #else:
        #st.write("Something is not right")
    return response.json(), response.status_code

def get_shap():
    url = 'http://localhost:8000/show-shap'
    response = requests.post(url)
    if response.status_code == 200:
        # Display the image
        image = Image.open(BytesIO(response.content))
        st.image(image, caption='SHAP Summary Plot')
    else:
        st.write("Failed to fetch the SHAP summary plot.")

def get_info_page():
    if 'is_valid' not in st.session_state:
        st.session_state.is_valid = [True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    if 'answers' not in st.session_state:
        st.session_state.answers = [None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    # Create a form
    st.markdown(
        """
        <style>
        .instruction {
            background-color: #f0f0f0; /* Light gray background */
            font-weight: bold;
            padding: 20px;
            border-radius: 5px; /* Curved edges */
            color: black;
            font-size: 20px;
            width: 100%
        }
        .stApp h2 {
            color: black;
            text-align: center;
            background-color: rgba(255, 255, 255, 0.8);
            margin-bottom: 8px;
        }
        .outcome {
            background-color: #f0f0f0;
            color: black;
            padding: 20px;
            border-radius: 5px;
            width: 100%;
        }
        .span {
            font-size: 25px;
            padding: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""
                <div class="instruction"> 
                    Please fill in all the details below: 
                </div>
                """,
                unsafe_allow_html = True)
    with st.form("my_form"):
        
        # Question 1 (Feature 1)
        acct_status_vals = ['', 'A11',           'A12'                                               , 'A13',                                                                      'A14']
        acct_status_desc = ['', "Less than 0 DM","Less than 200 DM but greater than or equal to 0 DM", "Greater than or equal to 200 DM / salary assignments for at least 1 year", "No checking account"]
        acct_status = st.selectbox('1. a) Checking Account Status', acct_status_desc,  format_func=lambda x: 'Select an option' if x == '' else x)
        acct_index = acct_status_vals[acct_status_desc.index(acct_status)]
        st.session_state.answers[0] = None if acct_index == '' else acct_index
        update_response(0)
        
        
        
        # Question 2 (Feature 2)
        credit_hist_vals = ['', 'A30',                                         'A31',                                     'A32',                                      'A33',                             'A34']
        credit_hist_desc = ['', "No credits taken/all credits paid back duly", "All credits at this bank paid back duly", "Existing credits paid back duly till now", "Delay in paying off in the past", "Critical account/other credits existing (not at this bank)"]
        credit_hist = st.selectbox('2. b) Credit History', credit_hist_desc, format_func=lambda x: 'Select an option' if x == '' else x)
        credit_hist_index = credit_hist_vals[credit_hist_desc.index(credit_hist)]
        st.session_state.answers[1] = None if credit_hist_index == '' else credit_hist_index
        update_response(1)
        
        
        
        # Question 3 (Feature 4)
        savings_vals = ['', 'A61',              'A62',                                               'A63',                                                'A64',                              'A65']
        savings_desc = ['', "Less than 100 DM", "Greater than or equal to 100 but less than 500 DM", "Greater than or equal to 500 but less than 1000 DM", "Greater than or equal to 1000 DM", "Unknown/no savings account"]
        savings = st.selectbox('3. c) Savings account/bonds', savings_desc,  format_func=lambda x: 'Select an option' if x == '' else x)
        savings_index = savings_vals[savings_desc.index(savings)]
        st.session_state.answers[2] = None if savings_index == '' else savings_index
        update_response(2)
            
        
        # Question 4 (Feature 5)
        emp_duration_vals = ['', 'A71',        'A72',              'A73',                                              'A74',                                              'A75']
        emp_duration_desc = ['', "Unemployed", "Less than 1 year", "Greater than or equal to 1 but less than 4 years", "Greater than or equal to 4 but less than 7 years", "Greater than or equal to 7 years"]
        emp_duration = st.selectbox('4. d) Employment Duration', emp_duration_desc, format_func=lambda x: 'Select an option' if x == '' else x)
        emp_duration_index = emp_duration_vals[emp_duration_desc.index(emp_duration)]
        st.session_state.answers[3] = None if emp_duration_index == '' else emp_duration_index
        update_response(3)
        
        
        # Question 5 (Feature 6)
        status_sex_vals =      ['', 'A91',                       'A92',                                 'A93',           'A94',                    'A95']
        status_sex_desc = ['', "Male : divorced/separated", "Female : divorced/separated/married", "Male : single", "Male : married/widowed", "Female : single"]
        status_sex = st.selectbox('5. e) Personal Status and Sex', status_sex_desc,  format_func=lambda x: 'Select an option' if x == '' else x)
        status_sex_index = status_sex_vals[status_sex_desc.index(status_sex)]
        st.session_state.answers[4] = None if status_sex_index == '' else status_sex_index
        update_response(4)
        
        # Question 6 (Feature 7)
        debt_gua_vals = ['', 'A101', 'A102',         'A103']
        debt_gua_desc = ['', 'None', 'Co-applicant', 'Guarantor']
        debt_gua = st.selectbox('6. f) Other Debtors/Guarantors', debt_gua_desc, format_func=lambda x: 'Select an option' if x == '' else x)
        debt_gua_index = debt_gua_vals[debt_gua_desc.index(debt_gua)]
        st.session_state.answers[5] = None if debt_gua_index == '' else debt_gua_index
        update_response(5)
        
        # Question 7 (Feature 8)
        instal_plans_vals = ['', 'A141', 'A142', 'A143']
        instal_plans_desc = ['', 'Bank', 'Store', 'None']
        instal_plans = st.selectbox('7. g) Other Instalment Plans', instal_plans_desc, format_func=lambda x: 'Select an option' if x == '' else x)
        instal_plans_index = instal_plans_vals[instal_plans_desc.index(instal_plans)]
        st.session_state.answers[6] = None if instal_plans_index == '' else instal_plans_index
        update_response(6)
            
                
        # Question 8 (Feature 10)                
        no_people_liable = st.number_input('8. h) Number of People being Liable to Provide Maintenance for',  value=None, min_value=None, max_value=None, step=None, placeholder="Enter value")
        st.session_state.answers[7] = None if no_people_liable is None else no_people_liable
        update_response(7)
        
        
        # Question 9 (Feature 11)
        telephone_vals = ['', 'A191', 'A192']
        telephone_desc = ['', 'None', 'Yes']
        telephone = st.selectbox('9. i) Has Telephone?', telephone_desc,  format_func=lambda x: 'Select an option' if x == '' else x)
        telephone_index = telephone_vals[telephone_desc.index(telephone)]
        st.session_state.answers[8] = None if telephone_index == '' else telephone_index
        update_response(8)
        
        
        # Question 10 (Feature 13)
        duration_vals = ["", "Short term", "Medium term", "Long term"]
        duration = st.selectbox('10. j) Duration', duration_vals,  format_func=lambda x: 'Select an option' if x == '' else x)
        duration_index = duration
        st.session_state.answers[9] = None if duration_index == '' else duration_index
        update_response(9)
        
        
        # Question 11 (Feature 3)
        purpose_vals = ["", "A40",       "A41",        "A42",                 "A43",              "A44",                 "A45",     "A46",       "A49",      "A410"]
        purpose_desc = ["", "Car (new)", "Car (used)", "Furniture/equipment", "Radio/television", "Domestic appliances", "Repairs", "Education", "Business", "Others"]
        purpose = st.selectbox('11. k) Purpose', purpose_desc,  format_func=lambda x: 'Select an option' if x == '' else x)
        purpose_index = purpose_vals[purpose_desc.index(purpose)]
        st.session_state.answers[10] = None if purpose_index == '' else purpose_index
        update_response(10)        
        
        
        # Question 12 (Feature 14)
        age_vals = ["", "20s and lower", "30s", "40s", "50s", "Seniors"]
        age = st.selectbox('12. l) Age', age_vals, format_func=lambda x: 'Select an option' if x == '' else x)
        age_index = age
        st.session_state.answers[11] = None if age_index == '' else age_index
        update_response(11)
        
        
        # Question 13 (Feature 12)
        is_foreign_vals = ['', 'A201', 'A202']
        is_foreign_desc = ['', 'Yes',  'No']
        is_foreign = st.selectbox('13. m) Is Foreign Worker?', is_foreign_desc, format_func=lambda x: 'Select an option' if x == '' else x)
        is_foreign_index = is_foreign_vals[is_foreign_desc.index(is_foreign)]
        st.session_state.answers[12] = None if is_foreign_index == '' else is_foreign_index
        update_response(12) 
        
        
        
        
        # Question 14 (Feature 9)
        existing_credits = st.number_input('14. n) Existing Credits at Bank',  value=None, min_value=None, max_value=None, step=None, placeholder="Enter value")
        st.session_state.answers[13] = None if existing_credits == '' else existing_credits
        update_response(13)
        
        # Add a checkbox
        agree = st.checkbox('I agree that all the information entered are correct')
        
        col1, col2, col3 = st.columns([1, 1, 0.5])
        with col2:
            # Add a submit button
            submitted = st.form_submit_button("PREDICT")
            #submitted = st.form_submit_button("PREDICT", type="secondary")
        # with col3:
        #     resetBtn = st.form_submit_button("Reset")
        # if resetBtn:
        #     st.rerun()
            
        
        if submitted:
            if agree:
                if check_response() == True:
                    st.success("Form submitted successfully!")
                    input_data = {
                        'checking_acct_status': st.session_state.answers[0],
                        'credit_history': st.session_state.answers[1],
                        'purpose': st.session_state.answers[10],
                        'savings_acct_bonds': st.session_state.answers[2],
                        'employment_duration': st.session_state.answers[3],
                        'personal_status_sex': st.session_state.answers[4],
                        'other_debtors_guarantors': st.session_state.answers[5],
                        'other_installment_plans': st.session_state.answers[6],
                        'existing_credits_at_bank': st.session_state.answers[13],
                        'no_of_people_liable': st.session_state.answers[7],
                        'telephone': st.session_state.answers[8],
                        'foreign_worker': st.session_state.answers[12],
                        'duration_category': st.session_state.answers[9],
                        'age_category': st.session_state.answers[11]
                    }
                    #for i in input_data.items():
                        #st.write(i[0], i[1])
                    with st.spinner('Making prediction...'):
                        prediction, status = get_prediction(input_data)
                    if status == 200:
                        st.markdown("---")
                        predicted_class = prediction['predictions'][0][0]
                    #p0_prob = prediction['predictions'][0][1]
                    #p1_prob = prediction['predictions'][0][2]
                    #plot_bytes = prediction.get("plot_bytes", None)
                        message = "Likely to DEFAULT" if predicted_class == 1 else "Not Likely to DEFAULT"
                        if predicted_class == 1:
                            st.markdown(f"""
                                    <div class="outcome">
                                        <h2>Outcome</h2>
                                        <b><span class="msg">Prediction      : <span style="color: red">{message}</span></span></b></br>
                                        <b><span class="msg">Suggested Action: <span style="color: red">Do not approve loan</span></span></b>
                                    </div>
                                    """, unsafe_allow_html = True
                                    )
                        else:
                            st.markdown(f"""
                                    <div class="outcome">
                                        <h2>Outcome</h2>
                                        <b><span class="msg">Prediction      : <span style="color: green">{message}</span></span></b></br>
                                        <b><span class="msg">Suggested Action: <span style="color: green">Approve loan</span></span></b>
                                    </div>
                                    """, unsafe_allow_html = True
                                    )
                        with st.spinner('Loading Explanation...'):
                            get_shap()
                    else:
                        st.markdown(f"""
                                <div class="outcome">
                                    <h2>Outcome</h2>
                                    <b><span class="msg"><span style="color: red">Server is down. Please try again later.</span></span></b></br>
                                </div>
                                """, unsafe_allow_html = True
                                )
                        
                    
                    #st.write(f"Probability of not defaulting: {p0_prob * 100:.2f}%")
                    #st.write(f"Probability of defaulting: {p1_prob * 100:.2f}%")
                    #if plot_bytes:
                        #img = Image.open(io.BytesIO(plot_bytes))
                        #st.write("SHAP Summary Plot:")
                        #st.image(img)
                    
                else:
                    st.error("Please answer all questions.")
                
            else:
                st.error("Please confirm the correctness of the information entered above.")
            for i in range(0, 14):
                st.session_state.is_valid[i] = st.session_state.answers[i] is not None
        
        

    css="""
    <style>
        [data-testid="stForm"] {
            background: LightBlue;
            }
        body {
            background-image: url("https://miro.medium.com/v2/resize:fit:1200/1*BFtMT2-yL6MkiWII4ySuqw.jpeg");
            background-size: contain;
            }
        .stSelectbox [data-testid='stMarkdownContainer'] {
            
            font-weight: bold;
            }
        [data-testid="stNumberInput"] {
            font-size: 20px;
            font-weight: bold;
            }
        </style>
        """
    st.write(css, unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()

# stApp {
#       background-image: url("https://miro.medium.com/v2/resize:fit:1200/1*BFtMT2-yL6MkiWII4ySuqw.jpeg");
#       background-size: contain;
#    }


