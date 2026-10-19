import os
import json
import joblib
import pandas as pd
import gradio as gr


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_RUN_ID = "470bd3cdf4e541d9946df8164e50b626"

ARTIFACTS_DIR = os.path.join(
    "artifacts",
    MODEL_RUN_ID
)

MODEL_PATH = os.path.join(
    ARTIFACTS_DIR,
    "model.pkl"
)

FEATURE_COLUMNS_PATH = os.path.join(
    ARTIFACTS_DIR,
    "feature_columns.json"
)

PREPROCESSING_PATH = os.path.join(
    ARTIFACTS_DIR,
    "preprocessing.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

with open(FEATURE_COLUMNS_PATH, "r") as f:
    FEATURE_COLUMNS = json.load(f)

# Loaded to keep the same artifact structure as your project
preprocessing_artifact = joblib.load(PREPROCESSING_PATH)

print("✅ XGBoost model loaded successfully")
print(f"✅ Loaded {len(FEATURE_COLUMNS)} feature columns")
print("✅ Preprocessing artifact loaded")


# ============================================================
# BINARY ENCODING
# ============================================================

BINARY_MAP = {
    "gender": {
        "Female": 0,
        "Male": 1
    },

    "Partner": {
        "No": 0,
        "Yes": 1
    },

    "Dependents": {
        "No": 0,
        "Yes": 1
    },

    "PhoneService": {
        "No": 0,
        "Yes": 1
    },

    "PaperlessBilling": {
        "No": 0,
        "Yes": 1
    }
}


# ============================================================
# NUMERIC COLUMNS
# ============================================================

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


# ============================================================
# CATEGORICAL COLUMNS
# ============================================================

CATEGORICAL_COLUMNS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod"
]


# ============================================================
# TRANSFORMATION
# ============================================================

def transform_for_prediction(customer_data):

    df = pd.DataFrame([customer_data])

    # Numeric columns
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            ).fillna(0)

    # Binary columns
    for column, mapping in BINARY_MAP.items():
        if column in df.columns:
            df[column] = (
                df[column]
                .astype(str)
                .str.strip()
                .map(mapping)
                .fillna(0)
                .astype(int)
            )

    # One-hot encoding
    df = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
        dtype=int
    )

    # Match training features exactly
    df = df.reindex(
        columns=FEATURE_COLUMNS,
        fill_value=0
    )

    df = df.astype(float)

    return df


# ============================================================
# PREDICTION
# ============================================================

def predict_churn(
    gender,
    Partner,
    Dependents,
    SeniorCitizen,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    tenure,
    MonthlyCharges,
    TotalCharges
):

    customer_data = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "SeniorCitizen": SeniorCitizen,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    X = transform_for_prediction(customer_data)

    probability = model.predict_proba(X)[0][1]

    threshold = 0.35

    prediction = int(probability >= threshold)

    result = (
        "⚠️ Likely to Churn"
        if prediction == 1
        else "✅ Not Likely to Churn"
    )

    return (
        prediction,
        f"{probability:.2%}",
        result
    )


# ============================================================
# GRADIO UI
# ============================================================

demo = gr.Interface(
    fn=predict_churn,

    inputs=[
        gr.Dropdown(
            ["Female", "Male"],
            label="Gender",
            value="Male"
        ),

        gr.Dropdown(
            ["No", "Yes"],
            label="Partner",
            value="No"
        ),

        gr.Dropdown(
            ["No", "Yes"],
            label="Dependents",
            value="No"
        ),

        gr.Number(
            label="Senior Citizen",
            value=0
        ),

        gr.Dropdown(
            ["No", "Yes"],
            label="Phone Service",
            value="Yes"
        ),

        gr.Dropdown(
            ["No phone service", "No", "Yes"],
            label="Multiple Lines",
            value="No"
        ),

        gr.Dropdown(
            ["DSL", "Fiber optic", "No"],
            label="Internet Service",
            value="Fiber optic"
        ),

        gr.Dropdown(
            ["No internet service", "No", "Yes"],
            label="Online Security",
            value="No"
        ),

        gr.Dropdown(
            ["No internet service", "No", "Yes"],
            label="Online Backup",
            value="No"
        ),

        gr.Dropdown(
            ["No internet service", "No", "Yes"],
            label="Device Protection",
            value="No"
        ),

        gr.Dropdown(
            ["No internet service", "No", "Yes"],
            label="Tech Support",
            value="No"
        ),

        gr.Dropdown(
            ["No internet service", "No", "Yes"],
            label="Streaming TV",
            value="No"
        ),

        gr.Dropdown(
            ["No internet service", "No", "Yes"],
            label="Streaming Movies",
            value="No"
        ),

        gr.Dropdown(
            ["Month-to-month", "One year", "Two year"],
            label="Contract",
            value="Month-to-month"
        ),

        gr.Dropdown(
            ["No", "Yes"],
            label="Paperless Billing",
            value="Yes"
        ),

        gr.Dropdown(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            label="Payment Method",
            value="Electronic check"
        ),

        gr.Number(
            label="Tenure (months)",
            value=1
        ),

        gr.Number(
            label="Monthly Charges",
            value=70
        ),

        gr.Number(
            label="Total Charges",
            value=70
        )
    ],

    outputs=[
        gr.Number(label="Prediction"),
        gr.Textbox(label="Churn Probability"),
        gr.Textbox(label="Result")
    ],

    title="📊 Telco Customer Churn Prediction",
    description=(
        "Predict whether a telecom customer is likely to churn "
        "using an XGBoost machine learning model."
    )
)


# ============================================================
# LAUNCH
# ============================================================

demo.launch()