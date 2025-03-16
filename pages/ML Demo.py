import streamlit as st
import numpy as np
import joblib

# โหลดโมเดล
model_1 = joblib.load("model/knn_model.pkl")
model_2 = joblib.load("model/svm_model.pkl")

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Lung Cancer Prediction", layout="centered")
st.title("🫁 Lung Cancer Prediction Form Input")

st.write("กรอกข้อมูลด้านล่างเพื่อทำนายโอกาสเป็นโรคมะเร็งปอด")

# สร้างฟอร์มรับค่าจากผู้ใช้
with st.form("user_input_form"):
    age = st.number_input("AGE (ปี)", min_value=1, max_value=120, step=1)
    gender = st.selectbox("GENDER", ["Male", "Female"])
    smoking = st.selectbox("SMOKING", ["Yes", "No"])
    finger_discolor = st.selectbox("FINGER DISCOLORATION", ["Yes", "No"])
    mental_stress = st.selectbox("MENTAL STRESS", ["Yes", "No"])
    pollution = st.selectbox("EXPOSURE TO POLLUTION", ["Yes", "No"])
    illness = st.selectbox("LONG TERM ILLNESS", ["Yes", "No"])
    energy = st.slider("ENERGY LEVEL", 10, 100, 50)
    immune = st.selectbox("IMMUNE WEAKNESS", ["Yes", "No"])
    breathing = st.selectbox("BREATHING ISSUE", ["Yes", "No"])
    alcohol = st.selectbox("ALCOHOL CONSUMPTION", ["Yes", "No"])
    throat = st.selectbox("THROAT DISCOMFORT", ["Yes", "No"])
    oxygen = st.slider("OXYGEN SATURATION (%)", 50, 100, 95)
    chest = st.selectbox("CHEST TIGHTNESS", ["Yes", "No"])
    family = st.selectbox("FAMILY HISTORY", ["Yes", "No"])
    smoking_family = st.selectbox("SMOKING FAMILY HISTORY", ["Yes", "No"])
    stress_immune = st.selectbox("STRESS IMPACT ON IMMUNE SYSTEM", ["Yes", "No"])
    
    submit = st.form_submit_button("🔍 ทำนายผล")

# แปลงค่าข้อมูลเป็นตัวเลขสำหรับโมเดล
if submit:
    input_data = np.array([[
        age, 
        1 if gender == "Male" else 0, 
        1 if smoking == "Yes" else 0,
        1 if finger_discolor == "Yes" else 0,
        1 if mental_stress == "Yes" else 0,
        1 if pollution == "Yes" else 0,
        1 if illness == "Yes" else 0,
        energy,
        1 if immune == "Yes" else 0,
        1 if breathing == "Yes" else 0,
        1 if alcohol == "Yes" else 0,
        1 if throat == "Yes" else 0,
        oxygen,
        1 if chest == "Yes" else 0,
        1 if family == "Yes" else 0,
        1 if smoking_family == "Yes" else 0,
        1 if stress_immune == "Yes" else 0
    ]], dtype=np.float32)
    
    # ทำการพยากรณ์จากทั้ง 2 โมเดล
    prediction_1 = model_1.predict(input_data)
    prediction_2 = model_2.predict(input_data)
    
    # แสดงผลลัพธ์
    st.write("🔍 ค่าที่โมเดล 1 ทำนาย:", "Yes" if prediction_1 == 1 else "No")
    st.write("🔍 ค่าที่โมเดล 2 ทำนาย:", "Yes" if prediction_2 == 1 else "No")