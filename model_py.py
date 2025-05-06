import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from fpdf import FPDF
from datetime import datetime
import base64

# --- Display Logo ---
logo = Image.open("ttu_logo.png")
st.image(logo, width=900)

# --- Header Info ---
st.markdown("""
<h2 style='text-align: center; color: navy;'>Graduation Project II</h2>
<h3 style='text-align: center; color: darkgreen;'>College of Engineering / Natural Resources and Chemical Engineering Department</h3>
<h4 style='text-align: center;'>Tafila Technical University</h4>
<h5 style='text-align: center; color: gray;'>Designed and implemented by students:</h5>
<ul style='text-align: center; list-style: none; padding-left: 0;'>
    <li>1 -Shahad Mohammad Abushamma</li>
    <li>2 - Rahaf Ramzi Al-shakh Qasem</li>
    <li>3 - Duaa Musa Al-Khalafat</li>
<h5 style='text-align: center; color: gray;'>Supervised by Dr.Ashraf Alsafasfeh</h5>
<ul style='text-align: center; list-style: none; padding-left: 0;'>

</ul>
""", unsafe_allow_html=True)

# --- Load ML Model and Scalers ---
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')
model = load_model('ann_water_model.h5')

# --- Separate Output Variables ---
water_quality_vars = [
    'Turbidity_final_NTU', 'Fe_final_mg_L', 'Mn_final_mg_L', 'Cu_final_mg_L',
    'Zn_final_mg_L', 'Suspended_solids_final_mg_L', 'TDS_final_mg_L',
    'Turbidity_removal_%', 'Suspended_solids_removal_%', 'TDS_removal_%'
]

operation_params_vars = [
    'Coagulant_dose_mg_L', 'Flocculant_dose_mg_L', 'Mixing_speed_rpm',
    'Rapid_mix_time_min', 'Slow_mix_time_min', 'Settling_time_min'
]

output_vars = [
    'Turbidity_final_NTU', 'Fe_final_mg_L', 'Mn_final_mg_L', 'Cu_final_mg_L',
    'Zn_final_mg_L', 'Suspended_solids_final_mg_L', 'TDS_final_mg_L',
    'Turbidity_removal_%', 'Suspended_solids_removal_%', 'TDS_removal_%','Coagulant_dose_mg_L',
    'Flocculant_dose_mg_L', 'Mixing_speed_rpm',
    'Rapid_mix_time_min', 'Slow_mix_time_min', 'Settling_time_min'
]

limits = {
    'Turbidity_final_NTU': {'op': '<=', 'value': 5.0},
    'Fe_final_mg_L': {'op': '<=', 'value': 0.3},
    'Mn_final_mg_L': {'op': '<=', 'value': 0.1},
    'Cu_final_mg_L': {'op': '<=', 'value': 1.0},
    'Zn_final_mg_L': {'op': '<=', 'value': 5.0},
    'Suspended_solids_final_mg_L': {'op': '<=', 'value': 50.0},
    'TDS_final_mg_L': {'op': '>', 'value': 1000.0}  
}

# --- Title & Form ---
st.title("Water Treatment Quality Predictor (ANN-based)")
st.markdown("Enter experimental values below to predict treated water quality and assess reuse suitability.")

with st.form("input_form"):
    pH_raw = st.slider("pH of Raw Water", 3.0, 11.0, 7.0)
    turbidity_raw = st.slider("Turbidity (NTU)", 0.1, 500.0, 50.0)
    temperature = st.slider("Temperature (°C)", 5.0, 40.0, 25.0)
    fe_initial = st.slider("Initial Fe (mg/L)", 0.0, 10.0, 1.0)
    mn_initial = st.slider("Initial Mn (mg/L)", 0.0, 5.0, 0.3)
    cu_initial = st.slider("Initial Cu (mg/L)", 0.0, 2.0, 0.05)
    zn_initial = st.slider("Initial Zn (mg/L)", 0.0, 5.0, 0.1)
    ss = st.slider("Suspended Solids (mg/L)", 0.0, 1000.0, 150.0)
    tds = st.slider("TDS (mg/L)", 0.0, 5000.0, 1000.0)

    submitted = st.form_submit_button("Test Water Quality")

# --- Prediction Logic ---
if submitted:
    input_array = np.array([[pH_raw, turbidity_raw, temperature,
                             fe_initial, mn_initial, cu_initial, zn_initial, ss, tds]])

    X_scaled = scaler_X.transform(input_array)
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]

    # --- Separate Output Variables ---
    water_quality_vars = [
        'Turbidity_final_NTU', 'Fe_final_mg_L', 'Mn_final_mg_L', 'Cu_final_mg_L',
        'Zn_final_mg_L', 'Suspended_solids_final_mg_L', 'TDS_final_mg_L',
        'Turbidity_removal_%', 'Suspended_solids_removal_%', 'TDS_removal_%'
    ]

    operation_params_vars = [
        'Coagulant_dose_mg_L', 'Flocculant_dose_mg_L', 'Mixing_speed_rpm',
        'Rapid_mix_time_min', 'Slow_mix_time_min', 'Settling_time_min'
    ]

    # Corresponding units
    units_dict = {
        'Turbidity_final_NTU': 'NTU',
        'Fe_final_mg_L': 'mg/L',
        'Mn_final_mg_L': 'mg/L',
        'Cu_final_mg_L': 'mg/L',
        'Zn_final_mg_L': 'mg/L',
        'Suspended_solids_final_mg_L': 'mg/L',
        'TDS_final_mg_L': 'mg/L',
        'Turbidity_removal_%': '%',
        'Suspended_solids_removal_%': '%',
        'TDS_removal_%': '%',
        'Coagulant_dose_mg_L': 'mg/L',
        'Flocculant_dose_mg_L': 'mg/L',
        'Mixing_speed_rpm': 'rpm',
        'Rapid_mix_time_min': 'min',
        'Slow_mix_time_min': 'min',
        'Settling_time_min': 'min'
    }
    # --- Create Operation Parameters Table ---
    op_df = pd.DataFrame(columns=["Operation Parameter", "Predicted Value", "Unit"])
    for var in operation_params_vars:
        idx = output_vars.index(var)
        value = y_pred[idx]
        unit = units_dict[var]
        op_df.loc[len(op_df)] = [var, round(value, 3), unit]
        
    # --- Create Water Quality Table ---
    quality_df = pd.DataFrame(columns=["Parameter", "Predicted Value", "Standard Limit", "Unit", "Assessment"])
    reuse_safe = True

    for var in water_quality_vars:
        idx = output_vars.index(var)
        value = y_pred[idx]
        unit = units_dict[var]
        
        if var in limits:
            _, limit_val = limits[var]
            assessment = "✅ OK" if value <= limit_val else "❌ Exceeds Limit"
            if value > limit_val:
                reuse_safe = False
            std_limit = limit_val
        else:
            std_limit = "--"
            assessment = "--"
    
        quality_df.loc[len(quality_df)] = [
            var,
            round(value, 3),
            std_limit,
            unit,
            assessment
        ]

    

    # --- Display Tables ---


    st.subheader("Operation Parameters Values")
    st.dataframe(op_df)

    st.subheader("Predicted Treated Water Quality")
    st.dataframe(quality_df)
    
    # --- Decision Message ---
    st.subheader("Reuse Decision")
    if reuse_safe:
        st.success("Water is safe for reuse or discharge.")
    else:
        st.error("Water does NOT meet quality standards for reuse.")

# --- PDF Report Generation ---
    def create_pdf(df, reuse_safe):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "Water Quality Prediction Report", ln=True, align='C')
    
        pdf.ln(10)
        for index, row in df.iterrows():
            # Ensure only ASCII characters are used in strings
            assessment_text = row['Assessment'].replace("✅", "OK").replace("❌", "Exceeds") if isinstance(row['Assessment'], str) else row['Assessment']
            pdf.cell(200, 10, f"{row['Parameter']}: {row['Predicted Value']} {row['Unit']} (Standard: {row['Standard Limit']}) --> {assessment_text}", ln=True)
    
        pdf.ln(10)
        decision_text = "Water is safe for reuse or discharge." if reuse_safe else "Water does NOT meet reuse standards."
        pdf.multi_cell(0, 10, f"Final Decision:\n{decision_text}")
    
        return pdf
    
    # Generate and download PDF
        pdf = create_pdf(quality_df, reuse_safe)
        pdf.output("report.pdf")
    
    with open("report.pdf", "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    
    href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="water_quality_report.pdf">📥 Download Report as PDF</a>'
    st.markdown(href, unsafe_allow_html=True)
