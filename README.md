# Customer Churn Prediction App 📈

This repository contains the code for a simple web application that predicts customer churn. The app is built in Python using Streamlit and `scikit-learn`.

The machine learning model (a Random Forest Classifier) is trained to predict whether a telecom customer is likely to churn ("True") or stay ("False") based on their account information.

---

## 🚀 Live Demo

This application is designed to be deployed on Streamlit Community Cloud.

**(Note: After you deploy your app, paste the public URL here!)**

---

## 📂 Files in this Repository

* **`app.py`**: The main Python script that creates and runs the Streamlit web application. It loads the model, displays the user interface for input, and shows the final prediction.
* **`churn_model.joblib`**: The pre-trained `scikit-learn` pipeline. This single file contains all the data preprocessing steps (scaling, encoding) and the trained Random Forest model.
* **`requirements.txt`**: A list of the Python libraries required to run this application (streamlit, pandas, scikit-learn, joblib).

---

## 🏃‍♀️ How to Run This App Locally

If you want to run this application on your own computer, follow these steps:

1.  **Clone this repository** to your local machine:
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create a virtual environment** (this is highly recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
    ```

3.  **Install the required libraries**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

A new tab should automatically open in your web browser showing the application.
