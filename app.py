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
st.header('Customer Information')

# We use columns to make the layout cleaner
col1, col2 = st.columns(2)

with col1:
    # --- Categorical Inputs ---
    st.subheader('Categorical Features')
    
    st_input = st.selectbox(
        'State (st)', 
        STATE_OPTIONS,
        help="The 2-letter abbreviation for the customer's state."
    )
    
    arcode_input = st.selectbox(
        'Area Code (arcode)', 
        AREA_CODE_OPTIONS,
        help="The customer's 3-digit telephone area code."
    )
    
    intplan_input = st.radio(
        'International Plan (intplan)', 
        PLAN_OPTIONS,
        help="Does the customer have an active international plan (yes/no)?"
    )
    
    voice_input = st.radio(
        'Voice Mail Plan (voice)', 
        VOICE_OPTIONS,
        help="Does the customer have an active voice mail plan (yes/no)?"
    )

with col2:
    # --- Numerical Inputs ---
    st.subheader('Numerical Features')
    
    acclen_input = st.number_input(
        'Account Length (acclen)', 
        min_value=0, 
        value=100,
        help="How long the customer has had their account (e.g., in days)."
    )
    
    nummailmes_input = st.number_input(
        'Number of Voicemail Messages (nummailmes)', 
        min_value=0, 
        value=0,
        help="Total number of voicemail messages in the customer's inbox."
    )
    
    ncsc_input = st.number_input(
        'Customer Service Calls (ncsc)', 
        min_value=0, 
        value=1,
        help="Number of calls made by the customer to customer service."
    )
    
    # --- Other Numerical Features (Hidden) ---
    # For a full app, you would create st.number_input() for these.
    # For this demo, we'll set them to 0.
    
    # We can use st.expander to hide these and make the app cleaner
    with st.expander("Show other numerical features (demo only)"):
        st.info("In a full app, you would add inputs for all 15 features. For this demo, the rest are set to 0.", icon="ℹ️")
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
    
    # Get the probabilities for each class
    prob_stay = prediction_proba[0][0]
    prob_churn = prediction_proba[0][1]

    # Display the main result (Churn or Stay)
    if prediction[0] == 1:
        st.error('**Result: This customer is LIKELY TO CHURN.**', icon="🚨")
    else:
        st.success('**Result: This customer is LIKELY TO STAY.**', icon="✅")

    # Add an expander to explain what the terms mean
    with st.expander("What do these results mean?"):
        st.info("""
            * **LIKELY TO STAY (Not Churn):** The model predicts this customer is satisfied and will likely remain with the service.
            * **LIKELY TO CHURN (Churn):** The model predicts this customer is at a high risk of canceling their service.
        """)
    
    st.write("---")

    # Display the detailed probabilities in columns
    st.subheader("Prediction Probabilities")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Stay Probability:**\n\n# {prob_stay * 100:.2f}%")
        
    with col2:
        st.warning(f"**Churn Probability:**\n\n# {prob_churn * 100:.2f}%")

    # Display the data that was used for the prediction
    st.write("---")
    st.subheader("Input Data Used for Prediction:")
    
    # THIS IS THE LINE THAT WAS MISSING
    st.dataframe(input_df) 

    # This expander should come AFTER the dataframe
    with st.expander("Show data dictionary for the input table"):
        st.markdown("""
            * **st**: State (Customer's 2-letter state abbreviation)
            * **acclen**: Account Length (How long the account has been active)
            * **arcode**: Area Code (The customer's 3-digit area code)
            * **intplan**: International Plan (yes/no)
            * **voice**: Voice Mail Plan (yes/no)
            * **nummailmes**: Number of Voicemail Messages
            * **tdmin**: Total Day Minutes
            * **tdcal**: Total Day Calls
            * **tdchar**: Total Day Charge
            * **temin**: Total Evening Minutes
            * **tecal**: Total Evening Calls
            * **tecahr**: Total Evening Charge
            * **tnmin**: Total Night Minutes
            * **tncal**: Total Night Calls
            * **tnchar**: Total Night Charge
            * **timin**: Total International Minutes
            * **tical**: Total International Calls
            * **tichar**: Total International Charge
            * **ncsc**: Number of Customer Service Calls
        """)
