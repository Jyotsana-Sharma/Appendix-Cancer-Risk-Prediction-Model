import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
import pandas as pd
import os

HERE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Appendix Cancer Risk Predictor",
    page_icon="🔬",
    layout="wide"
)

# ── Spark + model loading (cached so it only runs once) ───────────────────────

@st.cache_resource
def load_resources():
    # Java 17+ restricts javax.security.auth.Subject — these flags re-open it for Spark
    java_opts = (
        "--add-opens=java.base/javax.security.auth=ALL-UNNAMED "
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.nio=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
        "--add-opens=java.base/java.util=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED "
    )
    spark = SparkSession.builder \
        .appName("AppendixCancerDemo") \
        .config("spark.ui.showConsoleProgress", "false") \
        .config("spark.driver.extraJavaOptions", java_opts) \
        .config("spark.executor.extraJavaOptions", java_opts) \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    models = {
        "Logistic Regression":    PipelineModel.load(os.path.join(HERE, "logistic_regression_model")),
        "Random Forest":          PipelineModel.load(os.path.join(HERE, "random_forest_model")),
        "GBT Classifier":         PipelineModel.load(os.path.join(HERE, "gbt_model")),
        "Positive-Weighted RF":   PipelineModel.load(os.path.join(HERE, "rf_model_positive_weighted")),
    }
    return spark, models

# ── Schema matching what the pipeline was trained on ─────────────────────────

INPUT_SCHEMA = StructType([
    StructField("Country",                 StringType(),  True),
    StructField("Age",                     IntegerType(), True),
    StructField("Gender",                  StringType(),  True),
    StructField("BMI",                     DoubleType(),  True),
    StructField("Smoking_Status",          StringType(),  True),
    StructField("Alcohol_Consumption",     StringType(),  True),
    StructField("Family_History_Cancer",   StringType(),  True),
    StructField("Genetic_Mutations",       StringType(),  True),
    StructField("Chronic_Diseases",        StringType(),  True),
    StructField("Physical_Activity_Level", StringType(),  True),
    StructField("Diet_Type",               StringType(),  True),
    StructField("Radiation_Exposure",      StringType(),  True),
    StructField("Previous_Cancers",        StringType(),  True),
    StructField("Blood_Pressure",          IntegerType(), True),
    StructField("Cholesterol_Level",       IntegerType(), True),
    StructField("White_Blood_Cell_Count",  DoubleType(),  True),
    StructField("Red_Blood_Cell_Count",    DoubleType(),  True),
    StructField("Platelet_Count",          IntegerType(), True),
    StructField("Symptom_Severity",        StringType(),  True),
])

# ── Known training metrics (from Spark_Training_results_report.md) ────────────

TRAINING_METRICS = {
    "Logistic Regression":  {"AUC": 0.4969, "Accuracy": 0.5106, "F1": 0.5792, "Pos. Recall": 0.4763, "Train Time": "136s"},
    "Random Forest":        {"AUC": 0.5022, "Accuracy": 0.6631, "F1": 0.6975, "Pos. Recall": 0.2703, "Train Time": "975s"},
    "GBT Classifier":       {"AUC": 0.5004, "Accuracy": 0.5342, "F1": 0.6001, "Pos. Recall": 0.4568, "Train Time": "1139s"},
    "Positive-Weighted RF": {"AUC": 0.5022, "Accuracy": 0.6603, "F1": 0.6956, "Pos. Recall": 0.2724, "Train Time": "1029s"},
}

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🔬 Appendix Cancer Risk Predictor")
st.caption("PySpark MLlib — 4 models trained on 260,000 patient records")
st.divider()

with st.spinner("Starting Spark session and loading models (first run takes ~20s)…"):
    spark, models = load_resources()

st.success("Models loaded. Fill in patient details below and click **Predict**.")
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────

with st.form("patient_form"):
    st.subheader("Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        country = st.selectbox("Country", [
            "Argentina","Australia","Brazil","Canada","China","Egypt",
            "France","Germany","India","Indonesia","Italy","Japan",
            "Mexico","Netherlands","Norway","Poland","Russia",
            "Saudi Arabia","South Africa","South Korea","Spain",
            "Sweden","Turkey","UK","USA"
        ], index=24)
        age    = st.slider("Age", 18, 89, 45)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        bmi    = st.slider("BMI", 10.0, 50.0, 25.0, step=0.1)

    with col2:
        st.markdown("**Lifestyle & History**")
        smoking          = st.selectbox("Smoking Status",          ["No", "Yes"])
        alcohol          = st.selectbox("Alcohol Consumption",     ["Low", "Moderate", "High"])
        family_history   = st.selectbox("Family History of Cancer",["No", "Yes"])
        genetic          = st.selectbox("Genetic Mutations",       ["No", "Yes"])
        chronic_raw      = st.selectbox("Chronic Diseases",        ["None", "Diabetes", "Hypertension"])
        physical         = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
        diet             = st.selectbox("Diet Type",               ["Non-Vegetarian", "Vegetarian", "Vegan"])
        radiation        = st.selectbox("Radiation Exposure",      ["No", "Yes"])
        prev_cancers     = st.selectbox("Previous Cancers",        ["No", "Yes"])
        symptom_severity = st.selectbox("Symptom Severity",        ["Mild", "Moderate", "Severe"])

    with col3:
        st.markdown("**Clinical Measurements**")
        blood_pressure   = st.slider("Blood Pressure (mmHg)",      90,  179, 134)
        cholesterol      = st.slider("Cholesterol Level (mg/dL)",  150, 299, 224)
        wbc              = st.slider("WBC Count (×10³/µL)",        0.5, 13.7, 7.0, step=0.1)
        rbc              = st.slider("RBC Count (×10⁶/µL)",        2.8, 7.6,  5.0, step=0.1)
        platelet         = st.slider("Platelet Count (×10³/µL)",   150, 399, 274)

    submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────

if submitted:
    # "None" chronic disease was stored as empty string in training data
    chronic = "" if chronic_raw == "None" else chronic_raw

    row = [(
        country, age, gender, float(bmi), smoking, alcohol,
        family_history, genetic, chronic, physical, diet,
        radiation, prev_cancers, blood_pressure, cholesterol,
        float(wbc), float(rbc), platelet, symptom_severity
    )]

    input_df = spark.createDataFrame(row, schema=INPUT_SCHEMA)

    st.divider()
    st.subheader("Prediction Results")

    result_cols = st.columns(4)

    for (model_name, model), col in zip(models.items(), result_cols):
        try:
            pred = model.transform(input_df)
            row_out    = pred.select("prediction", "probability").first()
            prediction = int(row_out["prediction"])
            prob_pos   = float(row_out["probability"][1])

            label   = "⚠️ Cancer Risk" if prediction == 1 else "✅ Low Risk"
            color   = "red" if prediction == 1 else "green"
            metrics = TRAINING_METRICS[model_name]

            with col:
                st.markdown(f"**{model_name}**")
                st.markdown(
                    f"<h3 style='color:{color}'>{label}</h3>",
                    unsafe_allow_html=True
                )
                st.metric("Cancer probability", f"{prob_pos:.1%}")
                st.progress(prob_pos)

        except Exception as e:
            with col:
                st.error(f"{model_name} failed: {e}")

    # ── Model comparison table ────────────────────────────────────────────────

    st.divider()
    st.subheader("Model Performance on Test Set (260K dataset)")
    st.caption("Trained on 80% of data, evaluated on 51,840 records.")

    metrics_df = pd.DataFrame(TRAINING_METRICS).T.reset_index()
    metrics_df.columns = ["Model", "AUC", "Accuracy", "F1", "Positive Recall", "Train Time"]
    metrics_df["AUC"]             = metrics_df["AUC"].map("{:.4f}".format)
    metrics_df["Accuracy"]        = metrics_df["Accuracy"].map("{:.2%}".format)
    metrics_df["F1"]              = metrics_df["F1"].map("{:.4f}".format)
    metrics_df["Positive Recall"] = metrics_df["Positive Recall"].map("{:.2%}".format)

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.info(
        "**Note:** Positive Recall measures how many actual cancer cases the model catches. "
        "Logistic Regression and GBT have higher recall (47%, 46%) but also more false positives. "
        "Random Forest has the best overall F1 but only catches 27% of true cancer cases."
    )
