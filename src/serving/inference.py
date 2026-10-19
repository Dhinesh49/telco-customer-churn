import os
import json
import joblib
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)


# ============================================================
# MODEL RUN ID
# ============================================================

MODEL_RUN_ID = os.getenv("MODEL_RUN_ID")

if not MODEL_RUN_ID:
    raise ValueError(
        "MODEL_RUN_ID environment variable is not set. "
        "Example: MODEL_RUN_ID=470bd3cdf4e541d9946df8164e50b626"
    )


# ============================================================
# ARTIFACT DIRECTORY
# ============================================================

ARTIFACTS_DIR = os.path.join(
    PROJECT_ROOT,
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
# CHECK FILES
# ============================================================

for path in [
    MODEL_PATH,
    FEATURE_COLUMNS_PATH,
    PREPROCESSING_PATH
]:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required artifact not found: {path}"
        )


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)

print(
    f"✅ XGBoost model loaded successfully "
    f"(run: {MODEL_RUN_ID})"
)


# ============================================================
# LOAD FEATURE SCHEMA
# ============================================================

with open(
    FEATURE_COLUMNS_PATH,
    "r"
) as f:

    FEATURE_COLUMNS = json.load(f)


print(
    f"✅ Loaded {len(FEATURE_COLUMNS)} feature columns"
)


# ============================================================
# LOAD PREPROCESSING INFORMATION
# ============================================================

preprocessing_artifact = joblib.load(
    PREPROCESSING_PATH
)

print(
    "✅ Preprocessing artifact loaded"
)


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
    },
}


# ============================================================
# NUMERIC COLUMNS
# ============================================================

NUMERIC_COLUMNS = [

    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
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
    "PaymentMethod",
]


# ============================================================
# FEATURE TRANSFORMATION
# ============================================================

def transform_for_prediction(customer_data):

    df = pd.DataFrame(
        [customer_data]
    )


    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    for column in NUMERIC_COLUMNS:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            ).fillna(0)


    # --------------------------------------------------------
    # Binary columns
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # One-hot encoding
    # --------------------------------------------------------

    df = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
        dtype=int
    )


    # --------------------------------------------------------
    # Match training feature columns exactly
    # --------------------------------------------------------

    df = df.reindex(
        columns=FEATURE_COLUMNS,
        fill_value=0
    )


    # --------------------------------------------------------
    # Ensure numeric values
    # --------------------------------------------------------

    df = df.astype(float)


    return df


# ============================================================
# PREDICTION
# ============================================================

def predict(customer_data):

    X = transform_for_prediction(
        customer_data
    )


    # Probability of Churn
    probability = model.predict_proba(
        X
    )[0][1]


    # Classification threshold
    threshold = 0.35

    prediction = int(
        probability >= threshold
    )


    return {

        "prediction": prediction,

        "churn_probability": round(
            float(probability),
            4
        ),

        "result": (
            "Likely to Churn"
            if prediction == 1
            else "Not Likely to Churn"
        )
    }