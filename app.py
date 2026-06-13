import streamlit as st
import pandas as pd
import pickle
from sklearn.cluster import DBSCAN

# Load Model and Scaler
model = pickle.load(open("fraud_model (2).pkl", "rb"))
scaler = pickle.load(open("scaler (2).pkl", "rb"))

st.set_page_config(
    page_title="Digital Payment Fraud Detection",
    layout="wide"
)

st.title("💳 Digital Payment Fraud Detection")
st.subheader("Enter Transaction Details")

# Input Fields

step = st.number_input(
    "Step",
    min_value=1,
    value=1
)

transaction_type = st.selectbox(
    "Transaction Type",
    [
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER"
    ]
)

amount = st.number_input(
    "Amount",
    min_value=0.0,
    value=0.0
)

oldbalanceOrg = st.number_input(
    "Sender Old Balance",
    min_value=0.0,
    value=0.0
)

newbalanceOrig = st.number_input(
    "Sender New Balance",
    min_value=0.0,
    value=0.0
)

oldbalanceDest = st.number_input(
    "Receiver Old Balance",
    min_value=0.0,
    value=0.0
)

newbalanceDest = st.number_input(
    "Receiver New Balance",
    min_value=0.0,
    value=0.0
)

# Predict Button

if st.button("Check Fraud Risk"):

    transaction = {
        'step': step,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'isFlaggedFraud': 0,

        'type_CASH_IN': 0,
        'type_CASH_OUT': 0,
        'type_DEBIT': 0,
        'type_PAYMENT': 0,
        'type_TRANSFER': 0,
        'DBSCAN_Outlier': 0
    }

    # One Hot Encoding

    if transaction_type == "CASH_IN":
        transaction['type_CASH_IN'] = 1

    elif transaction_type == "CASH_OUT":
        transaction['type_CASH_OUT'] = 1

    elif transaction_type == "DEBIT":
        transaction['type_DEBIT'] = 1

    elif transaction_type == "PAYMENT":
        transaction['type_PAYMENT'] = 1

    elif transaction_type == "TRANSFER":
        transaction['type_TRANSFER'] = 1

    df = pd.DataFrame([transaction])

    feature_order = [
        'step',
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest',
        'isFlaggedFraud',
        'type_CASH_IN',
        'type_CASH_OUT',
        'type_DEBIT',
        'type_PAYMENT',
        'type_TRANSFER',
        'DBSCAN_Outlier'
    ]

    df = df[feature_order]

    scaled_data = scaler.transform(df)

    probability = model.predict_proba(
        scaled_data
    )[0][1]


    risk_score = probability * 100

    # Display Metrics

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Fraud Probability",
            f"{probability:.4f}"
        )

    with col2:
        st.metric(
            "Risk Score",
            f"{risk_score:.2f}"
        )

    # Risk Level

    if probability >= 0.75:
        risk_level = "HIGH"

    elif probability >= 0.25:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    st.write(f"### Risk Level: {risk_level}")

    # Fraud Detection Threshold

    FRAUD_THRESHOLD = 0.9999037824539824

    if probability >= FRAUD_THRESHOLD:

        st.error("🚨 FRAUD DETECTED")

    elif probability >= 0.75:

        st.warning(
            "⚠ HIGH RISK TRANSACTION - Review Recommended"
        )

    elif probability >= 0.25:

        st.info(
            "🟡 MEDIUM RISK TRANSACTION"
        )

    else:

        st.success(
            "🟢 LOW RISK TRANSACTION"
        )

    # Debug Information

    st.write("Raw Probability:", probability)