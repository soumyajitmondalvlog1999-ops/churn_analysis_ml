import streamlit as st
import pandas as pd
import joblib

# --- 1. LOAD THE TRAINED MODEL ---
# Use st.cache_resource to load the model only once and store it in cache
@st.cache_resource
def load_model():
    model = joblib.load('churn_model.joblib')
    return model

model = load_model()

# --- 2. DEFINE THE FEATURE NAMES (Must match the training data) ---
# These are the columns your model expects, in the correct order.
# We found these in Step 4 of your Colab notebook.
NUMERIC_FEATURES = [
    'acclen', 'nummailmes', 'tdmin', 'tdcal', 'tdchar',
    'temin', 'tecal', 'tecahr', 'tnmin', 'tncal', 'tnchar',
    'timin', 'tical', 'tichar', 'ncsc'
]

CATEGORICAL_FEATURES = ['st', 'arcode', 'intplan', 'voice']

# We also need the options for the dropdowns
# (You should get the full list of states from your data)
STATE_OPTIONS = ['KS', 'OH', 'NJ', 'GA', 'VT', 'MD', 'IN', '...'] # Add all 50+ states!
PLAN_OPTIONS = ['no', 'yes']
VOICE_OPTIONS = ['no', 'yes']
AREA_CODE_OPTIONS = ['415', '510', '408'] # From your data

# --- 3. CREATE THE APP INTERFACE ---
st.title('Customer Churn Prediction Model 📈')
st.write("""
Enter the customer's details below to predict whether they are likely to churn.
This model was trained on a public dataset and achieved ~98% accuracy.
""")

st.header('Customer Information')

# --- 4. CREATE INPUT WIDGETS ---
# We use columns to make the layout cleaner
col1, col2 = st.columns(2)

with col1:
    # --- Categorical Inputs ---
    st.subheader('Categorical Features')
    st_input = st.selectbox('State (st)', STATE_OPTIONS)
    arcode_input = st.selectbox('Area Code (arcode)', AREA_CODE_OPTIONS)
    intplan_input = st.radio('International Plan (intplan)', PLAN_OPTIONS)
    voice_input = st.radio('Voice Mail Plan (voice)', VOICE_OPTIONS)

with col2:
    # --- Numerical Inputs ---
    st.subheader('Numerical Features')
    acclen_input = st.number_input('Account Length (acclen)', min_value=0, value=100)
    nummailmes_input = st.number_input('Number of Voicemail Messages (nummailmes)', min_value=0, value=0)
    ncsc_input = st.number_input('Number of Customer Service Calls (ncsc)', min_value=0, value=1)
    
    # You can add the rest of the numerical features here if you want
    # For this example, I'm only adding a few key ones.
    # The model will use default values (0) for the rest.
    
    # Let's create dummy inputs for the rest so the model works
    # A more advanced app would have inputs for all of these.
    tdmin_input = 0
    tdcal_input = 0
    tdchar_input = 0
    temin_input = 0
    tecal_input = 0
    tecahr_input = 0
    tnmin_input = 0
    tncal_input = 0
    tnchar_input = 0
    timin_input = 0
    tical_input = 0
    tichar_input = 0
    
# --- 5. PREDICTION LOGIC ---
if st.button('**Predict Churn**', type="primary"):

    # Create a dictionary of all inputs
    input_data = {
        'acclen': acclen_input,
        'nummailmes': nummailmes_input,
        'tdmin': tdmin_input,
        'tdcal': tdcal_input,
        'tdchar': tdchar_input,
        'temin': temin_input,
        'tecal': tecal_input,
        'tecahr': tecahr_input,
        'tnmin': tnmin_input,
        'tncal': tncal_input,
        'tnchar': tnchar_input,
        'timin': timin_input,
        'tical': tical_input,
        'tichar': tichar_input,
        'ncsc': ncsc_input,
        'st': st_input,
        'arcode': arcode_input,
        'intplan': intplan_input,
        'voice': voice_input
    }

    # Create a DataFrame from the dictionary
    # The [0] is important to create a single-row DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Re-order columns to match the training data
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    input_df = input_df[all_features]
    
    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # --- 6. DISPLAY RESULT ---
    st.header('Prediction Result')
    
    if prediction[0] == 1:
        st.error('**Result: This customer is LIKELY TO CHURN.**', icon="🚨")
        st.write(f"Churn Probability: {prediction_proba[0][1] * 100:.2f}%")
    else:
        st.success('**Result: This customer is LIKELY TO STAY.**', icon="✅")
        st.write(f"Stay Probability: {prediction_proba[0][0] * 100:.2f}%")
        
    st.write("---")
    st.subheader("Input Data for Prediction:")
    st.dataframe(input_df)