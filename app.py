from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent


@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE / "models" / "xgb_model.pkl")
    cols = joblib.load(BASE / "models" / "feature_columns.pkl")
    return model, cols


model, cols = load_artifacts()


def risk_level(p):
    if p >= 0.60:
        return "High"
    elif p >= 0.30:
        return "Medium"
    return "Low"


def build_features(i):
    """Turn form inputs into the exact columns the model was trained on."""
    yn = lambda v: 1 if v == "Yes" else 0
    internet = i["InternetService"]
    no_net = internet == "No"
    phone = i["PhoneService"] == "Yes"

    # No internet / no phone means the add-on services are "No"
    multi = yn(i["MultipleLines"]) if phone else 0
    addons = {k: (0 if no_net else yn(i[k])) for k in
              ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
               "TechSupport", "StreamingTV", "StreamingMovies"]}

    tenure, monthly, total = i["tenure"], i["MonthlyCharges"], i["TotalCharges"]
    num_services = (yn(i["PhoneService"]) + multi + sum(addons.values())
                    + (0 if no_net else 1))

    if tenure <= 12:
        group = "0-12"
    elif tenure <= 24:
        group = "13-24"
    elif tenure <= 48:
        group = "25-48"
    else:
        group = "49-72"

    row = {
        "gender": 1 if i["gender"] == "Male" else 0,
        "SeniorCitizen": yn(i["SeniorCitizen"]),
        "Partner": yn(i["Partner"]),
        "Dependents": yn(i["Dependents"]),
        "tenure": tenure,
        "PhoneService": yn(i["PhoneService"]),
        "MultipleLines": multi,
        **addons,
        "PaperlessBilling": yn(i["PaperlessBilling"]),
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "num_services": num_services,
        "auto_pay": 1 if "automatic" in i["PaymentMethod"] else 0,
        "has_protection": 1 if (addons["OnlineSecurity"] or addons["TechSupport"]) else 0,
        "avg_monthly_spend": total / tenure if tenure > 0 else monthly,
        "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
        "InternetService_No": 1 if no_net else 0,
        "Contract_One year": 1 if i["Contract"] == "One year" else 0,
        "Contract_Two year": 1 if i["Contract"] == "Two year" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if i["PaymentMethod"] == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if i["PaymentMethod"] == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if i["PaymentMethod"] == "Mailed check" else 0,
        "tenure_group_13-24": 1 if group == "13-24" else 0,
        "tenure_group_25-48": 1 if group == "25-48" else 0,
        "tenure_group_49-72": 1 if group == "49-72" else 0,
    }
    return pd.DataFrame([row]).reindex(columns=cols, fill_value=0)


RULES = [
    {
        "action": "Offer discounted annual plan",
        "offer": "15% off the monthly price for moving to a 1-year contract",
        "reason": "Month-to-month customer with high monthly charges",
        "test": lambda r: r.Contract == "Month-to-month" and r.MonthlyCharges >= 70,
    },
    {
        "action": "Contract upgrade incentive",
        "offer": "One free month or a device credit for moving to a 1-year contract",
        "reason": "Month-to-month customers can leave at any time",
        "test": lambda r: r.Contract == "Month-to-month",
    },
    {
        "action": "Free tech support trial",
        "offer": "3 months of free tech support",
        "reason": "Internet customer without tech support",
        "test": lambda r: r.InternetService != "No" and r.TechSupport == "No",
    },
    {
        "action": "Free online security trial",
        "offer": "3 months of free online security",
        "reason": "Internet customer without online security",
        "test": lambda r: r.InternetService != "No" and r.OnlineSecurity == "No",
    },
    {
        "action": "Move to auto-pay",
        "offer": "A 5 dollar bill credit for switching to automatic payment",
        "reason": "Pays by electronic check, which has high churn",
        "test": lambda r: r.PaymentMethod == "Electronic check",
    },
    {
        "action": "Fiber loyalty discount",
        "offer": "A loyalty discount on the fiber plan plus a speed check call",
        "reason": "High-paying fiber customer",
        "test": lambda r: r.InternetService == "Fiber optic" and r.MonthlyCharges >= 80,
    },
    {
        "action": "New customer check-in",
        "offer": "A welcome call and a setup help session in the first year",
        "reason": "Customer is in the first 12 months, when churn is highest",
        "test": lambda r: r.tenure <= 12,
    },
    {
        "action": "Senior support plan",
        "offer": "Priority phone support and a simplified bill",
        "reason": "Senior customer at risk of leaving",
        "test": lambda r: r.SeniorCitizen == 1,
    },
]

DEFAULT = {
    "action": "Loyalty reward",
    "offer": "A thank-you credit on the next bill",
    "reason": "At-risk customer with no specific trigger",
}


def suggest_actions(i, level):
    # Same as the notebook: actions are only recommended for High and Medium risk
    if level == "Low":
        return ["No urgent action. Keep monitoring and reward loyalty."]

    r = SimpleNamespace(
        Contract=i["Contract"],
        MonthlyCharges=i["MonthlyCharges"],
        InternetService=i["InternetService"],
        TechSupport=i["TechSupport"],
        OnlineSecurity=i["OnlineSecurity"],
        PaymentMethod=i["PaymentMethod"],
        tenure=i["tenure"],
        SeniorCitizen=1 if i["SeniorCitizen"] == "Yes" else 0,
    )
    matches = [rule for rule in RULES if rule["test"](r)]
    main = matches[0] if matches else DEFAULT
    out = [f"**{main['action']}**: {main['offer']} ({main['reason']})"]
    if len(matches) > 1:
        second = matches[1]
        out.append(f"Also consider: {second['action']}. {second['offer']}")
    return out


st.set_page_config(page_title="Churn Risk Predictor", page_icon="📉")
st.title("Customer Churn Risk Predictor")
st.caption("Enter a customer's details to estimate their chance of leaving.")

with st.form("customer"):
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox("Payment method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, step=1.0)
        total = st.number_input("Total charges ($), 0 to estimate", 0.0, 10000.0, 0.0, step=10.0)
    with c2:
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multi = st.selectbox("Multiple lines", ["No", "Yes"])
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        security = st.selectbox("Online security", ["No", "Yes"])
        backup = st.selectbox("Online backup", ["No", "Yes"])
        device = st.selectbox("Device protection", ["No", "Yes"])
        tech = st.selectbox("Tech support", ["No", "Yes"])
        tv = st.selectbox("Streaming TV", ["No", "Yes"])
        movies = st.selectbox("Streaming movies", ["No", "Yes"])
    submitted = st.form_submit_button("Predict churn risk")

if submitted:
    if total == 0:
        total = float(tenure * monthly)  # estimate when not provided
    inputs = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multi, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": device, "TechSupport": tech,
        "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }
    prob = float(model.predict_proba(build_features(inputs))[0, 1])
    level = risk_level(prob)

    st.subheader("Result")
    m1, m2 = st.columns(2)
    m1.metric("Churn probability", f"{prob:.1%}")
    m2.metric("Risk level", level)
    st.progress(min(prob, 1.0))

    st.subheader("Suggested actions")
    for a in suggest_actions(inputs, level):
        st.write(f"- {a}")
        
