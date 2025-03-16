import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np

# โหลดโมเดลที่ผ่านการเทรนแล้ว
try:
    model = tf.keras.models.load_model('model/cnn_model.keras')
except OSError:
    st.error("ไม่พบไฟล์โมเดล 'model/cnn_model.keras'. โปรดตรวจสอบพาธของไฟล์.")
    st.stop()

# ฟังก์ชันสำหรับการทำนายภาพ
def predict_image(image):
    # ปรับขนาดภาพให้ตรงกับที่โมเดลคาดหวัง (146x146)
    image = image.resize((146, 146))
    image_array = np.array(image) / 255.0  # ปรับค่าให้อยู่ในช่วง [0,1]
    image_array = np.expand_dims(image_array, axis=0)  # เพิ่มมิติให้เป็น (1, 146, 146, 3)

    # ทำนายผล
    predictions = model.predict(image_array)
    return 'แมว' if np.argmax(predictions) == 0 else 'หมา'

# หน้าแอป Streamlit
st.title("ทำนายภาพหมาหรือแมว 🐶🐱")
st.write("อัปโหลดภาพของหมาหรือแมว แล้วแอปจะทำนายว่าเป็นหมาหรือแมว")

# อัปโหลดไฟล์ภาพ
uploaded_file = st.file_uploader("อัปโหลดภาพ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='ภาพที่อัปโหลด', use_container_width=True)

    # ทำนายผล
    prediction = predict_image(image)
    st.write(f"🔎 **ผลการทำนาย:** {prediction}")