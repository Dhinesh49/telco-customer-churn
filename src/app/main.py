"""
FASTAPI + GRADIO SERVING APPLICATION
====================================

Telco Customer Churn Prediction

- FastAPI → REST API
- Gradio → Web UI
- Pydantic → Request validation
- inference.py → Model prediction
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import gradio as gr

from src.serving.inference import predict


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML API for predicting customer churn",
    version="1.0.0"
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    """
    Health check endpoint.
    """
    return {
        "status": "ok"
    }


# ============================================================
# REQUEST DATA SCHEMA
# ============================================================

class CustomerData(BaseModel):
    """
    Customer data schema.

    These are the 19 features required for prediction.
    """

    # --------------------------------------------------------
    # Demographics
    # --------------------------------------------------------

    gender: str
    Partner: str
    Dependents: str
    SeniorCitizen: int


    # --------------------------------------------------------
    # Phone services
    # --------------------------------------------------------

    PhoneService: str
    MultipleLines: str


    # --------------------------------------------------------
    # Internet services
    # --------------------------------------------------------

    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str


    # --------------------------------------------------------
    # Account information
    # --------------------------------------------------------

    Contract: str
    PaperlessBilling: str
    PaymentMethod: str


    # --------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------

    tenure: int
    MonthlyCharges: float
    TotalCharges: float


# ============================================================
# PREDICTION API
# ============================================================

@app.post("/predict")
def get_prediction(data: CustomerData):
    """
    Predict customer churn.

    Receives customer data,
    sends it to inference.py,
    and returns the prediction.
    """

    try:

        # Convert Pydantic object to dictionary
        customer_data = data.model_dump()

        # Call ML inference
        result = predict(
            customer_data
        )

        # Return prediction directly
        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# GRADIO INTERFACE FUNCTION
# ============================================================

def gradio_interface(
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
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

    """
    Gradio interface.

    Converts UI inputs into the same format
    used by the FastAPI prediction endpoint.
    """

    # --------------------------------------------------------
    # Create customer dictionary
    # --------------------------------------------------------

    data = {

        "gender": gender,

        "SeniorCitizen": int(
            SeniorCitizen
        ),

        "Partner": Partner,

        "Dependents": Dependents,

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

        "tenure": int(
            tenure
        ),

        "MonthlyCharges": float(
            MonthlyCharges
        ),

        "TotalCharges": float(
            TotalCharges
        )
    }


    # --------------------------------------------------------
    # Call same inference function
    # --------------------------------------------------------

    result = predict(
        data
    )


    # Return result to Gradio
    return str(result)


# ============================================================
# GRADIO UI
# ============================================================

demo = gr.Interface(

    fn=gradio_interface,

    inputs=[

        # ----------------------------------------------------
        # Demographics
        # ----------------------------------------------------

        gr.Dropdown(
            ["Male", "Female"],
            label="Gender",
            value="Male"
        ),

        gr.Number(
            label="Senior Citizen (0 = No, 1 = Yes)",
            value=0,
            minimum=0,
            maximum=1,
            precision=0
        ),

        gr.Dropdown(
            ["Yes", "No"],
            label="Partner",
            value="No"
        ),

        gr.Dropdown(
            ["Yes", "No"],
            label="Dependents",
            value="No"
        ),


        # ----------------------------------------------------
        # Phone services
        # ----------------------------------------------------

        gr.Dropdown(
            ["Yes", "No"],
            label="Phone Service",
            value="Yes"
        ),

        gr.Dropdown(
            ["Yes", "No", "No phone service"],
            label="Multiple Lines",
            value="No"
        ),


        # ----------------------------------------------------
        # Internet services
        # ----------------------------------------------------

        gr.Dropdown(
            ["DSL", "Fiber optic", "No"],
            label="Internet Service",
            value="Fiber optic"
        ),

        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Online Security",
            value="No"
        ),

        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Online Backup",
            value="No"
        ),

        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Device Protection",
            value="No"
        ),

        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Tech Support",
            value="No"
        ),

        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Streaming TV",
            value="Yes"
        ),

        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Streaming Movies",
            value="Yes"
        ),


        # ----------------------------------------------------
        # Contract and billing
        # ----------------------------------------------------

        gr.Dropdown(
            ["Month-to-month", "One year", "Two year"],
            label="Contract",
            value="Month-to-month"
        ),

        gr.Dropdown(
            ["Yes", "No"],
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


        # ----------------------------------------------------
        # Numeric features
        # ----------------------------------------------------

        gr.Number(
            label="Tenure (months)",
            value=1,
            minimum=0,
            maximum=100
        ),

        gr.Number(
            label="Monthly Charges ($)",
            value=85.0,
            minimum=0,
            maximum=200
        ),

        gr.Number(
            label="Total Charges ($)",
            value=85.0,
            minimum=0,
            maximum=10000
        )
    ],


    # ========================================================
    # OUTPUT
    # ========================================================

    outputs=gr.Textbox(
        label="Churn Prediction",
        lines=3
    ),


    # ========================================================
    # UI TITLE
    # ========================================================

    title="🔮 Telco Customer Churn Predictor",

    description="""
    **Predict customer churn probability using machine learning**

    Fill in the customer details below to get a churn prediction.

    The model uses XGBoost trained on historical telecom
    customer data.
    """,


    # ========================================================
    # EXAMPLES
    # ========================================================

    examples=[

        # High churn risk example
        [
            "Female",
            0,
            "No",
            "No",
            "Yes",
            "No",
            "Fiber optic",
            "No",
            "No",
            "No",
            "No",
            "Yes",
            "Yes",
            "Month-to-month",
            "Yes",
            "Electronic check",
            1,
            85.0,
            85.0
        ],

        # Low churn risk example
        [
            "Male",
            0,
            "Yes",
            "Yes",
            "Yes",
            "Yes",
            "DSL",
            "Yes",
            "Yes",
            "Yes",
            "Yes",
            "No",
            "No",
            "Two year",
            "No",
            "Credit card (automatic)",
            60,
            45.0,
            2700.0
        ]
    ],

    theme=gr.themes.Soft()
)


# ============================================================
# MOUNT GRADIO INTO FASTAPI
# ============================================================

app = gr.mount_gradio_app(
    app,
    demo,
    path="/ui"
)