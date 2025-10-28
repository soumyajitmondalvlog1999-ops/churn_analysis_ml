import streamlit as st
import pandas as pd
import numpy as np  # Make sure numpy is imported
import joblib

# --- 1. LOAD THE TRAINED MODEL ---
@st.cache_resource
def load_model():
    model = joblib.load('churn_model.joblib')
    return model

model = load_model()

# --- 2. DEFINE THE FEATURE NAMES (Must match the training data) ---
NUMERIC_FEATURES = [
    'acclen', 'nummailmes', 'tdmin', 'tdcal', 'tdchar',
    'temin', 'tecal', 'tecahr', 'tnmin', 'tncal', 'tnchar',
    'timin', 'tical', 'tichar', 'ncsc'
]

CATEGORICAL_FEATURES = ['st', 'arcode', 'intplan', 'voice']

# --- OPTIONS FOR DROPDOWNS ---
STATE_OPTIONS = ['KS', 'OH', 'NJ', 'GA', 'VT', 'MD', 'IN', 'NY', 'LA', 'AZ', 'MO', 'VA', 'OR', 'UT', 'WY', 'MI', 'ID', 'MT', 'MN', 'AL', 'WA', 'TX', 'RI', 'WI', 'CO', 'NV', 'MS', 'WV', 'KY', 'AR', 'DC', 'CT', 'HI', 'ME', 'NE', 'NM', 'NH', 'NC', 'SC', 'TN', 'OK', 'FL', 'IA', 'PA', 'SD', 'MA', 'CA', 'DE', 'IL', 'AK', 'ND'] # Added more states
PLAN_OPTIONS = ['no', 'yes']
VOICE_OPTIONS = ['no', 'yes']
AREA_CODE_OPTIONS = ['415', '510', '408']

# -------------------------------------------------------------------
# Main App Interface
# -------------------------------------------------------------------
st.title('📈 Customer Churn Prediction Model')
st.write("""
Enter the customer's details below to predict whether they are likely to churn.
This model was trained on a public dataset and achieved ~98% accuracy.
""")

# --- FEATURE IMPORTANCE SECTION (MODIFIED) ---
st.subheader("💡 What Factors Does the Model Use?")

try:
    # Access components from the loaded pipeline
    classifier = model.named_steps['classifier']
    preprocessor = model.named_steps['preprocessor']

    # Get importances and feature names
    importances = classifier.feature_importances_
    feature_names = preprocessor.get_feature_names_out()

    # Create a DataFrame
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # --- NEW: Calculate Percentage Importance ---
    imp_df['Importance (%)'] = (imp_df['Importance'] * 100).round(2)

    # Prepare DataFrame for the bar chart (index needed)
    chart_df = imp_df.set_index('Feature')

    st.info("""
    This chart shows the importance of each feature as a **percentage**.
    Higher percentages mean that feature has a bigger impact on the prediction.
    """, icon="ℹ️")

    # Display the bar chart (plotting the percentage column)
    st.bar_chart(chart_df['Importance (%)'].head(15))

    # Expander now shows both raw and percentage values
    with st.expander("Show all feature importances (raw and percentage)"):
        # Display the DataFrame with all columns
        st.dataframe(imp_df[['Feature', 'Importance', 'Importance (%)']].reset_index(drop=True))

except Exception as e:
    st.error(f"Could not load feature importance: {e}", icon="🚨")
    st.write("This can happen if the 'churn_model.joblib' file is not a pipeline or is from a different scikit-learn version.")
# --- END MODIFIED SECTION ---


# --- 4. CREATE INPUT WIDGETS ---
st.header('🔬 Enter Customer Information')

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
    with st.expander("Show all numerical features (optional)"):
        tdmin_input = st.number_input("Total Day Minutes", value=0.0)
        tdcal_input = st.number_input("Total Day Calls", value=0)
        tdchar_input = st.number_input("Total Day Charge", value=0.0)
        temin_input = st.number_input("Total Evening Minutes", value=0.0)
        tecal_input = st.number_input("Total Evening Calls", value=0)
        tecahr_input = st.number_input("Total Evening Charge", value=0.0)
        tnmin_input = st.number_input("Total Night Minutes", value=0.0)
        tncal_input = st.number_input("Total Night Calls", value=0)
        tnchar_input = st.number_input("Total Night Charge", value=0.0)
        timin_input = st.number_input("Total International Minutes", value=0.0)
        tical_input = st.number_input("Total International Calls", value=0)
        tichar_input = st.number_input("Total International Charge", value=0.0)


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
    input_df = pd.DataFrame([input_data])

    # Re-order columns to match the training data
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    input_df = input_df[all_features]

    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # --- 6. DISPLAY RESULT ---
    st.header('Prediction Result')

    prob_stay = prediction_proba[0][0]
    prob_churn = prediction_proba[0][1]

    if prediction[0] == 1:
        st.error('**Result: This customer is LIKELY TO CHURN.**', icon="🚨")
    else:
        st.success('**Result: This customer is LIKELY TO STAY.**', icon="✅")

    with st.expander("What do these results mean?"):
        st.info("""
            * **LIKELY TO STAY (Not Churn):** The model predicts this customer is satisfied and will likely remain with the service.
            * **LIKELY TO CHURN (Churn):** The model predicts this customer is at a high risk of canceling their service.
        """)

    st.write("---")

    st.subheader("Prediction Probabilities")
    col1, col2 = st.columns(2)

    with col1:
        st.info(f"**Stay Probability:**\n\n# {prob_stay * 100:.2f}%")

    with col2:
        st.warning(f"**Churn Probability:**\n\n# {prob_churn * 100:.2f}%")

    st.write("---")
    st.subheader("Input Data Used for Prediction:")
    st.dataframe(input_df)

    with st.expander("Show data dictionary for the input table"):
        st.markdown("""
            * **st**: State
            * **acclen**: Account Length
            * **arcode**: Area Code
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
            * **tncal**:Total Night Calls
            * **tnchar**: Total Night Charge
            * **timin**: Total International Minutes
            * **tical**: Total International Calls
            * **tichar**: Total International Charge
            * **ncsc**: Number of Customer Service Calls
        """)
