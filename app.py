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


# ------------------------- Page and design -------------------------
st.set_page_config(page_title="Churn Risk Predictor", page_icon="📉", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero {background: linear-gradient(135deg, #1e3a8a, #3b82f6); padding: 28px 32px;
       border-radius: 16px; color: white; margin-bottom: 24px;}
.hero h1 {color: white; margin: 0; font-size: 2rem;}
.hero p {margin: 6px 0 0; opacity: .9;}
.result-card {border-radius: 16px; padding: 28px 20px; color: white; text-align: center;}
.result-card .label {font-size: .9rem; opacity: .9; letter-spacing: .05em; text-transform: uppercase;}
.result-card .big {font-size: 3.6rem; font-weight: 700; line-height: 1.1; margin: 6px 0;}
.result-card .badge {display: inline-block; padding: 4px 16px; border-radius: 999px;
                     background: rgba(255,255,255,.22); font-weight: 600;}
.bar-bg {background: rgba(255,255,255,.3); border-radius: 999px; height: 10px; margin-top: 18px;}
.bar-fill {background: white; height: 10px; border-radius: 999px;}
.action {background: #f8fafc; border-left: 5px solid #3b82f6; padding: 12px 16px;
         border-radius: 8px; margin-bottom: 10px; color: #0f172a;}
.section-title {font-weight: 600; font-size: 1.05rem; margin-bottom: 4px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📉 Customer Churn Risk Predictor</h1>
  <p>Enter a customer's details to see their chance of leaving and what to do about it.</p>
</div>
""", unsafe_allow_html=True)

with st.form("customer"):
    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            st.markdown('<div class="section-title">👤 Customer</div>', unsafe_allow_html=True)
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

    with c2:
        with st.container(border=True):
            st.markdown('<div class="section-title">📡 Services</div>', unsafe_allow_html=True)
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multi = st.selectbox("Multiple lines", ["No", "Yes"])
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            security = st.selectbox("Online security", ["No", "Yes"])
            backup = st.selectbox("Online backup", ["No", "Yes"])
            device = st.selectbox("Device protection", ["No", "Yes"])
            tech = st.selectbox("Tech support", ["No", "Yes"])
            tv = st.selectbox("Streaming TV", ["No", "Yes"])
            movies = st.selectbox("Streaming movies", ["No", "Yes"])

    with c3:
        with st.container(border=True):
            st.markdown('<div class="section-title">💳 Billing</div>', unsafe_allow_html=True)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox("Payment method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, step=1.0)
            total = st.number_input("Total charges ($), 0 to estimate", 0.0, 10000.0, 0.0, step=10.0)

    submitted = st.form_submit_button("Predict churn risk", type="primary",
                                      use_container_width=True)

if not submitted:
    st.info("Fill in the details above and click **Predict churn risk**.")
else:
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
    color = {"High": "#dc2626", "Medium": "#f59e0b", "Low": "#16a34a"}[level]

    left, right = st.columns([1, 1.4])
    with left:
        st.markdown(f"""
        <div class="result-card" style="background:{color};">
          <div class="label">Churn probability</div>
          <div class="big">{prob:.1%}</div>
          <div class="badge">{level} risk</div>
          <div class="bar-bg"><div class="bar-fill" style="width:{min(prob, 1.0) * 100:.0f}%"></div></div>
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("#### Suggested actions")
        for a in suggest_actions(inputs, level):
            a = a.replace("**", "<b>", 1).replace("**", "</b>", 1)
            st.markdown(f'<div class="action">{a}</div>', unsafe_allow_html=True)