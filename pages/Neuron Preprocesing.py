import streamlit as st
import os
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Neuron Preprocessing", layout="wide")

# หัวข้อหน้า
st.title("🧠 Neuron Preprocessing - การเตรียมข้อมูลสำหรับ Neural Network")

st.write("การเตรียมข้อมูลเป็นขั้นตอนสำคัญในการพัฒนาโมเดล Neural Network เพื่อให้ได้ข้อมูลที่เหมาะสมสำหรับการเรียนรู้")

# สร้าง Tabs
tab1, tab2 = st.tabs(["🔍 ตรวจสอบข้อมูล & การเตรียมข้อมูล", "📝 อธิบายการ Train Model CNN"])

with tab1:
    # โฟลเดอร์ที่เก็บรูปภาพ
    base_folder = "data/animal/"
    categories = ["cat", "dog"]
    
    data_summary = []
    file_extensions = set()
    
    st.subheader("📋 รายละเอียดข้อมูลรูปภาพ")
    
    for category in categories:
        total_count = 0
        for dataset in ["train", "val"]:
            folder_path = os.path.join(base_folder, dataset, category)
            if os.path.exists(folder_path):
                images = [f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))]
                total_count += len(images)
                file_extensions.update([f.split('.')[-1].upper() for f in images])
        data_summary.append([category, f"~{total_count} รูป", "/".join(file_extensions)])
    
    # สร้าง DataFrame สำหรับแสดงข้อมูล
    df = pd.DataFrame(data_summary, columns=["ชื่อคลาส", "จำนวนรูป", "รูปแบบไฟล์"])
    st.table(df)

    st.subheader("🔄 การเตรียมข้อมูล")
    st.write("ใช้ ImageDataGenerator เพื่อทำ Data Augmentation และเตรียมข้อมูลให้เหมาะสมกับการฝึกโมเดล CNN")
    code_preprocess = """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)
    """
    st.code(code_preprocess, language="python")
    st.write("โค้ดด้านบนใช้สำหรับทำ Data Augmentation ในชุดข้อมูลฝึก และ Normalization ในชุดข้อมูล validation")
  
with tab2:
    st.subheader("📝 อธิบายการ Train Model CNN")
    
    st.write("### 📌 1. สร้างโมเดล CNN")
    code_cnn_1 = """
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    """
    st.code(code_cnn_1, language="python")
    st.write("โมเดล CNN นี้ประกอบไปด้วย Convolutional Layers, MaxPooling Layers, และ Fully Connected Layers")
    
    st.write("### 📌 2. คอมไพล์และเทรนโมเดล")
    code_cnn_2 = """
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=15
    )
    """
    st.code(code_cnn_2, language="python")
    st.write("คอมไพล์โมเดลด้วย loss function และ optimizer จากนั้นทำการเทรนโมเดลด้วยข้อมูลที่เตรียมไว้")

st.write("---")
st.write("📌 **Datasetจาก : www.kaggle.com**")