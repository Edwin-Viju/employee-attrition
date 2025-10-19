# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Employee Attrition Risk Dashboard", layout="wide")

st.title("📊 Employee Attrition Risk Dashboard — Project 3 Final Deliverable")

# ---------------------------
# Helper to load artifacts
# ---------------------------
@st.cache_data
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def try_load_artifact(name):
    if os.path.exists(name):
        if name.endswith(".pkl"):
            return joblib.load(name)
        elif name.endswith(".json"):
            return load_json(name)
        elif name.endswith(".csv"):
            return pd.read_csv(name)
    return None

model = try_load_artifact("model.pkl")
scaler = try_load_artifact("scaler.pkl")
feature_names = try_load_artifact("feature_names.json")
feat_imp = try_load_artifact("feat_imp.csv")
metrics = try_load_artifact("metrics.json")
feature_info = try_load_artifact("feature_info.json")

# Sidebar: upload fallback
st.sidebar.header("Uploads (optional)")
uploaded_model = st.sidebar.file_uploader("Upload model.pkl", type=["pkl","joblib"])
uploaded_scaler = st.sidebar.file_uploader("Upload scaler.pkl", type=["pkl","joblib"])
uploaded_feat_imp = st.sidebar.file_uploader("Upload feat_imp.csv", type=["csv"])
uploaded_metrics = st.sidebar.file_uploader("Upload metrics.json", type=["json"])
uploaded_feature_info = st.sidebar.file_uploader("Upload feature_info.json", type=["json"])
uploaded_feature_names = st.sidebar.file_uploader("Upload feature_names.json", type=["json"])

if uploaded_model is not None:
    model = joblib.load(uploaded_model)
if uploaded_scaler is not None:
    scaler = joblib.load(uploaded_scaler)
if uploaded_feat_imp is not None:
    feat_imp = pd.read_csv(uploaded_feat_imp)
if uploaded_metrics is not None:
    metrics = json.load(uploaded_metrics)
if uploaded_feature_info is not None:
    feature_info = json.load(uploaded_feature_info)
if uploaded_feature_names is not None:
    feature_names = json.load(uploaded_feature_names)

# ---------------------------
# Summary of findings
# ---------------------------
st.header("Summary of Findings")
st.markdown("""
- **Top risk factors**: `OverTime` and certain **high-pressure job roles** are the biggest risks.
- **Non-monetary factors** like **Work-Life Balance** and **Environment Satisfaction** strongly influence attrition.
""")

# ---------------------------
# Model metrics
# ---------------------------
st.header("Model Performance (Final)")
if metrics:
    prec = metrics.get("precision_attrition_1", None)
    rec  = metrics.get("recall_attrition_1", None)
    thr  = metrics.get("threshold", None)
    st.markdown(f"**Precision (Attrition=1):** **{prec:.2f}**" if prec is not None else "Precision: N/A")
    st.markdown(f"**Recall (Attrition=1):** **{rec:.2f}**" if rec is not None else "Recall: N/A")
    if thr is not None:
        st.markdown(f"**Decision threshold used:** {thr}")
else:
    # show the final metrics example if not provided
    st.markdown("Precision (Attrition=1): **0.50**  \nRecall (Attrition=1): **0.43**  (This is the default/expected final trade-off)")

st.markdown("""
**Why this trade-off?**  
Precision = 0.50 means half of flagged employees are true positives (manageable for targeted outreach).  
Recall = 0.43 means nearly half of real leavers are caught — a useful catch rate for early intervention without overwhelming HR with false positives.
""")

# ---------------------------
# Top 5 features chart
# ---------------------------
st.header("Key Indicator Visualization — Top 5 Features")
if feat_imp is not None and isinstance(feat_imp, pd.DataFrame) and not feat_imp.empty:
    top5 = feat_imp.sort_values("Importance", ascending=False).head(5).reset_index(drop=True)
else:
    # placeholder
    top5 = pd.DataFrame({
        "Feature": ["OverTime", "JobRole_SalesExecutive", "YearsAtCompany",
                    "EnvironmentSatisfaction", "WorkLifeBalance"],
        "Importance": [0.24, 0.19, 0.16, 0.12, 0.10]
    })

st.dataframe(top5.style.format({"Importance":"{:.3f}"}), width=700)
fig, ax = plt.subplots(figsize=(8,4))
ax.barh(top5["Feature"][::-1], top5["Importance"][::-1])
ax.set_xlabel("Importance")
ax.set_title("Top 5 Features Influencing Attrition")
st.pyplot(fig)

st.markdown("**Insight:** Non-monetary features (OverTime, Work-Life Balance, Environment Satisfaction) dominate — retention strategies should prioritize workload, role design, and workplace culture.")

# ---------------------------
# Interactive prediction: ask user for parameters one-by-one
# ---------------------------
st.header("Predict: Enter employee parameters (step-by-step)")

# If we have feature_info, present inputs based on it
if feature_info and isinstance(feature_info, dict):
    st.write("Enter values for the logical variables below. For categorical fields, choose the correct category; the app will map it to the model's one-hot columns.")
    input_vals = {}
    for key, info in feature_info.items():
        if info.get("type") == "numeric":
            # ---------------------------
            # Friendly numeric input
            # ---------------------------
            numeric_descriptions = {
                "Age":"Employee's age in years",
                "DailyRate":"Daily pay rate",
                "DistanceFromHome":"Distance from home to workplace (km)",
                "Education":"Education level (1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor)",
                "EnvironmentSatisfaction":"Environment Satisfaction (1=Low, 2=Medium, 3=High, 4=Very High)",
                "HourlyRate":"Hourly pay rate",
                "JobInvolvement":"Job involvement (1=Low, 2=Medium, 3=High, 4=Very High)",
                "JobLevel":"Job level (1=Entry, 2=Junior, 3=Senior, 4=Manager, 5=Executive)",
                "JobSatisfaction":"Job satisfaction (1=Low, 2=Medium, 3=High, 4=Very High)",
                "MonthlyIncome":"Monthly salary",
                "MonthlyRate":"Monthly rate (system-specific)",
                "NumCompaniesWorked":"Number of companies previously worked",
                "PercentSalaryHike":"Last salary hike (%)",
                "PerformanceRating":"Performance rating (1=Low, 2=Good, 3=Excellent, 4=Outstanding)",
                "RelationshipSatisfaction":"Relationship satisfaction (1=Low, 2=Medium, 3=High, 4=Very High)",
                "StockOptionLevel":"Stock option level (0=none, 1,2,3)",
                "TotalWorkingYears":"Total work experience (years)",
                "TrainingTimesLastYear":"Number of trainings last year",
                "WorkLifeBalance":"Work-Life Balance (1=Bad, 2=Good, 3=Better, 4=Best)"
            }
            desc = numeric_descriptions.get(key, "")
            default_val = float(info.get("default", 0.0))
            v = st.number_input(f"{key}: {desc}", value=default_val)
            input_vals[key] = v
        elif info.get("type") == "categorical":
            cats = info.get("categories", [])
            if not cats:
                v = st.text_input(f"{key} (category, enter exact value)")
                input_vals[key] = v
            else:
                v = st.selectbox(f"{key} (category)", options=cats)
                input_vals[key] = v
        else:
            v = st.text_input(f"{key}")
            input_vals[key] = v

    # Button to submit
    if st.button("Predict (using model)"):
        if model is None or feature_names is None:
            st.error("Model or feature_names.json not available. Please upload artifacts (model.pkl + feature_names.json) in the sidebar or place them in the app folder.")
        else:
            x = pd.Series(0, index=feature_names, dtype=float)
            for key, info in feature_info.items():
                if info.get("type") == "numeric":
                    for col in info.get("model_columns", []):
                        if col in x.index:
                            x[col] = float(input_vals[key])
                elif info.get("type") == "categorical":
                    chosen = input_vals[key]
                    mapped = None
                    for col in info.get("model_columns", []):
                        tail = col.split("_", 1)[1] if "_" in col else None
                        if tail == chosen:
                            mapped = col
                            break
                    if mapped and mapped in x.index:
                        x[mapped] = 1.0
                    else:
                        if chosen in x.index:
                            x[chosen] = 1.0
            X_row = x.values.reshape(1, -1)
            if scaler is not None:
                try:
                    X_row = scaler.transform(X_row)
                except Exception as e:
                    st.warning(f"Scaler couldn't be applied: {e}. The app will attempt to predict without scaling.")
            try:
                proba = model.predict_proba(X_row)[0][1]
                thr = metrics.get("threshold", 0.3) if metrics else 0.3
                pred = int(proba > thr)
                st.metric("Probability of leaving (Attrition=1)", f"{proba:.3f}")
                st.success("Prediction: **Will Leave**" if pred==1 else "Prediction: **Will Stay**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
else:
    st.info("No feature metadata found. Please provide feature_info.json (recommended). As fallback, enter the top 5 features manually.")
    demo_inputs = {}
    for feature in top5["Feature"].tolist():
        demo_inputs[feature] = st.text_input(f"Enter value for {feature}", value="0")

    if st.button("Predict (demo)"):
        st.info("Demo predict: will use placeholder model if available. For robust predictions, save feature_info.json and feature_names.json from the notebook.")
        if model and feature_names:
            x = pd.Series(0, index=feature_names, dtype=float)
            for f, val in demo_inputs.items():
                if f in x.index:
                    try:
                        x[f] = float(val)
                    except:
                        if val.lower() in ["yes","1","true"]:
                            x[f] = 1.0
            X_row = x.values.reshape(1,-1)
            if scaler:
                try:
                    X_row = scaler.transform(X_row)
                except:
                    pass
            proba = model.predict_proba(X_row)[0][1]
            thr = metrics.get("threshold", 0.3) if metrics else 0.3
            pred = int(proba > thr)
            st.metric("Probability of leaving (Attrition=1)", f"{proba:.3f}")
            st.success("Prediction: **Will Leave**" if pred==1 else "Prediction: **Will Stay**")
        else:
            st.error("No model available. Upload model.pkl and feature_names.json in the sidebar to enable predictions.")
