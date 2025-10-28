import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib
import openpyxl  # Required for pd.read_excel

# -------------------------------------------------------------------
# Page Setup
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Automated ML Model Builder",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Automated ML Model Builder")
st.write("Upload your data, select what you want to predict, and this app will automatically build a model for you.")

# -------------------------------------------------------------------
# build_model Function (Contains all the ML logic)
# -------------------------------------------------------------------
def build_model(df, target_column, problem_type, ignore_cols):
    """Builds and evaluates a model based on user inputs."""
    
    st.subheader("📊 1. Preparing Data")
    
    # --- 1. Define Features (X) and Target (y) ---
    try:
        y = df[target_column]
        X = df.drop(columns=[target_column] + ignore_cols)
    except Exception as e:
        st.error(f"Error preparing data: {e}", icon="🚨")
        return

    # --- 2. Sanity Checks & Target Variable Processing ---
    if problem_type == "Classification":
        y = y.astype(str).str.strip().str.replace('.', '', regex=False)
        unique_classes = y.nunique()
        if unique_classes > 50:
            st.error(f"""
            **Configuration Error!**
            
            You selected **Classification**, but your target column ('{target_column}')
            has **{unique_classes}** unique values.
            
            * Classification is for predicting a few categories (e.g., 'True'/'False').
            * Your data looks like a **Regression** problem (predicting a number).
            
            **Please select a different target column or change the "Problem Type" to "Regression".**
            """, icon="🚨")
            return
        if unique_classes < 2:
            st.error(f"""
            **Configuration Error!**
            
            Your target column ('{target_column}') only has **{unique_classes}** unique value.
            The model needs at least two different classes to learn from (e.g., 'True' and 'False').
            """, icon="🚨")
            return
            
        # If all checks pass, encode y for classification
        y_categories = y.astype('category').cat.categories
        y = y.astype('category').cat.codes
            
    else: # Problem type is "Regression"
        try:
            # Try to force y to be numeric for regression
            y = pd.to_numeric(y)
            
            if y.isna().any():
                st.warning(f"Your target column ('{target_column}') has missing values. Rows with missing targets will be dropped before training.", icon="⚠️")
                X = X[~y.isna()]
                y = y.dropna()
                
        except (ValueError, TypeError):
            st.error(f"""
            **Configuration Error!**
            
            You selected **Regression**, but your target column ('{target_column}')
            contains non-numeric text values (like '{y.iloc[0]}').
            
            * Regression can only predict numbers.
            * If you are trying to predict a category, please change the "Problem Type" to "Classification".
            """, icon="🚨")
            return

    # --- 3. Automatic Feature Detection & Cleaning (for X) ---
    st.subheader("🧬 2. Detecting Feature Types")
    
    numeric_features = []
    categorical_features = []
    
    for col in X.columns:
        try:
            pd.to_numeric(X[col])
            numeric_features.append(col)
        except (ValueError, TypeError):
            categorical_features.append(col)
            X[col] = X[col].astype(str) # Force to string to handle mixed types

    st.info(f"**Found {len(numeric_features)} numerical features:**\n{numeric_features}", icon="🔢")
    st.info(f"**Found {len(categorical_features)} categorical features:**\n{categorical_features}", icon="🔤")

    # --- 5. Create Preprocessing Pipelines ---
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # --- 6. Combine transformers ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough')

    # --- 7. Select Model Based on Problem Type ---
    if problem_type == "Classification":
        model = RandomForestClassifier(random_state=42)
    else: 
        model = RandomForestRegressor(random_state=42)

    # --- 8. Create the full ML Pipeline ---
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # --- 9. Split data and train model ---
    st.subheader("🛠️ 3. Training Model")
    with st.spinner("Splitting data and training model... This may take a moment."):
        
        try:
            stratify_option = (y if problem_type == 'Classification' else None)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify_option
            )
        except ValueError as e:
            st.warning(f"""
            **Stratification Warning:** Could not stratify data (Error: {e}). 
            This usually happens if one class has very few samples.
            Proceeding without stratification.
            """, icon="⚠️")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)
        
    st.success("Model training complete!", icon="✅")

    # --- 10. NEW: FEATURE IMPORTANCE SECTION ---
    st.subheader("💡 4. What Factors Mattered Most?")
    
    try:
        # Access the fitted model and preprocessor
        model = model_pipeline.named_steps['model']
        preprocessor = model_pipeline.named_steps['preprocessor']

        # Get feature importances
        importances = model.feature_importances_
        
        # Get feature names from the preprocessor
        feature_names = preprocessor.get_feature_names_out()
        
        # Create a DataFrame for easy plotting
        imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Set feature as index for st.bar_chart
        imp_df = imp_df.set_index('Feature')

        st.info(f"""
        This chart shows the most important factors the model used to make its predictions. 
        Higher values mean the feature had a bigger impact on the outcome.
        
        **Top 3 Factors:**
        1. {imp_df.index[0]}
        2. {imp_df.index[1]}
        3. {imp_df.index[2]}
        """, icon="ℹ️")

        # Display the bar chart (showing top 15 for clarity)
        st.bar_chart(imp_df.head(15))
        
        # Add an expander for the full list
        with st.expander("Show all feature importances"):
            st.dataframe(imp_df)
            
    except AttributeError:
        st.warning("Could not retrieve feature importances for this model type.", icon="⚠️")
    except Exception as e:
        st.error(f"An error occurred while calculating feature importance: {e}", icon="🚨")
    # --- END NEW SECTION ---

    # --- 11. Evaluate Model and Show Results (Re-numbered) ---
    st.subheader("📈 5. Model Evaluation")
    if problem_type == "Classification":
        accuracy = accuracy_score(y_test, y_pred)
        st.metric(label="**Accuracy**", value=f"{accuracy * 100:.2f}%")
        st.text("Classification Report:")
        try:
            st.code(classification_report(y_test, y_pred, target_names=y_categories.astype(str), zero_division=0))
        except Exception as e:
            st.warning(f"Could not generate classification report. (Error: {e})", icon="⚠️")
            st.code(classification_report(y_test, y_pred, zero_division=0))
            
    else: # Regression
        mse = mean_squared_error(y_test, y_pred) # Get MSE
        rmse = np.sqrt(mse) # Get RMSE by taking the square root
        r2 = r2_score(y_test, y_pred)
        
        st.metric(label="**R-squared (R2)**", value=f"{r2:.4f}")
        st.metric(label="**Root Mean Squared Error (RMSE)**", value=f"{rmse:.4f}")

    # --- 12. Sample Predictions (Re-numbered) ---
    st.subheader("📋 6. Sample Predictions")
    
    if problem_type == "Classification":
        y_test_labels = y_test.map(lambda x: y_categories[x])
        y_pred_labels = [y_categories[p] for p in y_pred]
        results_df = pd.DataFrame({'Actual': y_test_labels, 'Predicted': y_pred_labels})
    else:
        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        
    st.dataframe(results_df.head(10))

# -------------------------------------------------------------------
# Main App Interface
# -------------------------------------------------------------------

# --- 1. File Uploader ---
st.header("📤 1. Upload Your Data")
uploaded_file = st.file_uploader(
    "Upload your data (CSV, TXT, Excel)", 
    type=["csv", "xlsx", "xls", "txt"]
)

df = None  # Initialize df as None

if uploaded_file is not None:
    
    # --- 2. Load Options ---
    st.subheader("⚙️ Load Options")
    
    is_excel = uploaded_file.name.endswith(('.xls', '.xlsx'))
    
    # --- HEADER OPTIONS (VISIBLE FOR ALL FILES) ---
    no_header = st.checkbox(
        "My file does NOT have a header row", 
        value=False,
        help="Check this if your file *only* contains data and no column names. The app will use numbers (0, 1, 2...) as names."
    )
    
    if no_header:
        header_arg = None
    else:
        # If the file DOES have a header, let the user specify which row
        header_arg = st.number_input(
            "Header Row Number", 
            min_value=0, 
            value=0, 
            step=1,
            help="Which row contains the column names? (Remember: 0 is the first row)"
        )
        st.caption("E.g., enter **0** if the header is on the **first row**, or **2** if it's on the **third row**.")
    # --- END HEADER OPTIONS ---

    if is_excel:
        try:
            df = pd.read_excel(uploaded_file, header=header_arg)
        except Exception as e:
            st.error(f"Error loading Excel file: {e}. Check your header row number.", icon="🚨")
    else:
        # It's a text file (CSV, TXT, etc.)
        st.info("""
            **Please specify how to load your text file:**
            * **Header:** Use the options above to specify your header row.
            * **Delimiter:** What character separates your columns?
            """, icon="ℹ️")
        
        separator = st.text_input(
            "Delimiter (Separator)", 
            value=",", 
            help="What character separates your columns? (e.g., `,` for comma, `\\t` for tab, `;` for semicolon)"
        )
        
        if separator == "\\t": # Handle tab character
            separator = "\t"

        try:
            df = pd.read_csv(uploaded_file, sep=separator, header=header_arg)
        except Exception as e:
            st.error(f"Error loading text file: {e}. Check your delimiter and header option.", icon="🚨")

    # --- 3. Continue if 'df' is loaded ---
    if df is not None:
        st.success(f"File '{uploaded_file.name}' loaded successfully. Found {df.shape[0]} rows and {df.shape[1]} columns.", icon="✅")
        st.dataframe(df.head())
        
        # --- 4. User Configuration ---
        st.header("🔬 2. Configure Your Model")
        
        all_columns = df.columns.tolist()
        
        target_column = st.selectbox(
            "Which column do you want to predict? (Target Variable)",
            options=all_columns,
            index=None,
            help="Select the column the model should learn to predict."
        )
        
        problem_type = st.selectbox(
            "What type of problem is this?",
            options=["Classification", "Regression"],
            help="**Classification:** Predicts a category (e.q., 'Churn'/'Stay', 'Red'/'Green').\n\n**Regression:** predicts a number (e.g., price, temperature)."
        )

        ignore_cols = st.multiselect(
            "Select any columns to ignore (e.g., ID, Name, Phone Number)",
            options=[col for col in all_columns if col != target_column],
            help="Select columns that are not useful for prediction (like 'phnum' in your churn data)."
        )

        # --- 5. Run Button ---
        st.header("🚀 3. Build Your Model")
        if st.button("**Build Your Model**", type="primary"):
            if target_column is None:
                st.error("Please select a target variable to predict.", icon="🚨")
            else:
                build_model(df, target_column, problem_type, ignore_cols)
