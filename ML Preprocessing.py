import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Preprocessing", layout="wide")

st.title("📊 ML Data Preprocessing")

st.write("การเตรียมข้อมูลเป็นขั้นตอนสำคัญในการพัฒนาโมเดล Machine Learning และ Neural Network เพื่อให้ได้ข้อมูลที่เหมาะสมสำหรับการเรียนรู้")

tab1, tab2, tab3 = st.tabs(["🔍 ตรวจสอบข้อมูล", "📝 อธิบายการ Train Model SVM", "📝 อธิบายการ Train Model KNN"])

with tab1:
    data_file = "data/Lung_Cancer_Dataset_Corrupted.csv"
    
    if data_file is not None:
        df = pd.read_csv(data_file)
        
        st.subheader("📌 ตัวอย่างข้อมูลที่อัปโหลด")
        st.dataframe(df.head())
        
        st.subheader("❌ ตรวจสอบค่าที่หายไป")
        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0])
        
        st.subheader("🚨 ตรวจสอบค่า Outliers")
        numeric_columns = df.select_dtypes(include=['number']).columns
        outliers = {}
        
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            if outlier_count > 0:
                outliers[col] = outlier_count
        
        if outliers:
            st.write(outliers)
        else:
            st.write("✅ ไม่พบ Outliers ในชุดข้อมูล")

with tab2:
    st.subheader("📝 อธิบายการ Train Model SVM")
    
    st.write("### 📌 1. นำเข้าไลบรารีที่จำเป็น")
    code1 = """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    """
    st.code(code1, language="python")
    st.write("ส่วนนี้เป็นการนำเข้าไลบรารีที่จำเป็นสำหรับการทำ Machine Learning เช่น Pandas, NumPy, Scikit-learn")
    
    st.write("### 📌 2. โหลดข้อมูลและจัดการค่าที่หายไป")
    code2 = """
    file_path = "../data/Lung_Cancer_Dataset_Corrupted.csv"
    df = pd.read_csv(file_path)
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    """
    st.code(code2, language="python")
    st.write("โหลดข้อมูลจากไฟล์ CSV และจัดการค่าที่หายไปโดยใช้ค่ามัธยฐาน (Median) หรือค่าที่พบบ่อยที่สุด (Mode)")
    
    st.write("### 📌 3. จัดการค่า Outliers")
    code3 = """
    for col in df.select_dtypes(include=[np.number]).columns:
        mean = df[col].mean()
        std = df[col].std()
        lower_bound, upper_bound = mean - 3*std, mean + 3*std
        df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), mean, df[col])
    """
    st.code(code3, language="python")
    st.write("ใช้ค่าเฉลี่ย (Mean) และส่วนเบี่ยงเบนมาตรฐาน (Standard Deviation) เพื่อลดผลกระทบของ Outliers")
    
    st.write("### 📌 4. แปลงข้อมูลให้อยู่ในรูปแบบที่เหมาะสม")
    code4 = """
    last_col = df.columns[-1]
    df[last_col] = df[last_col].map({'YES': 1, 'NO': 0})
    df = df.astype(int)
    """
    st.code(code4, language="python")
    st.write("แปลงค่าหมวดหมู่ (Categorical Data) ให้เป็นตัวเลข เพื่อให้โมเดลสามารถเรียนรู้ได้")
    
    st.write("### 📌 5. แบ่งข้อมูลเป็นชุด Train และ Test")
    code5 = """
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    """
    st.code(code5, language="python")
    st.write("แบ่งข้อมูลออกเป็นชุด Train (80%) และ Test (20%)")
    
    st.write("### 📌 6. ทำการ Scaling ข้อมูล")
    code6 = """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    """
    st.code(code6, language="python")
    st.write("ใช้ StandardScaler เพื่อปรับขนาดข้อมูลให้อยู่ในช่วงเดียวกัน")
    
    st.write("### 📌 7. สร้างและฝึกโมเดล SVM")
    code7 = """
    model = SVC(kernel='linear', C=1.0, random_state=42)
    model.fit(X_train_scaled, y_train)
    """
    st.code(code7, language="python")
    st.write("สร้างโมเดล SVM โดยใช้ Kernel แบบ Linear และทำการ Train โมเดลด้วยข้อมูลที่เตรียมไว้")
    
    st.write("### 📌 8. ทดสอบโมเดล")
    code8 = """
    accuracy = model.score(X_test_scaled, y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    """
    st.code(code8, language="python")
    st.write("ทดสอบโมเดลโดยใช้ชุดข้อมูล Test และคำนวณค่าความแม่นยำ")

with tab3:
    st.subheader("📝 อธิบายการ Train Model KNN")
    
    st.write("### 📌 1. นำเข้าไลบรารีที่จำเป็น")
    code_knn_1 = """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    """
    st.code(code_knn_1, language="python")
    st.write("ส่วนนี้เป็นการนำเข้าไลบรารีที่จำเป็น เช่น Pandas, NumPy, Scikit-learn และ KNeighborsClassifier สำหรับโมเดล KNN")
    
    st.write("### 📌 2. โหลดข้อมูลและจัดการค่าที่หายไป")
    code_knn_2 = """
    file_path = "../data/Lung_Cancer_Dataset_Corrupted.csv"
    df = pd.read_csv(file_path)
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    """
    st.code(code_knn_2, language="python")
    st.write("โหลดข้อมูลจากไฟล์ CSV และจัดการค่าที่หายไปโดยใช้ค่ามัธยฐาน (Median) หรือค่าที่พบบ่อยที่สุด (Mode)")
    
    st.write("### 📌 3. แบ่งข้อมูลเป็นชุด Train และ Test")
    code_knn_3 = """
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    """
    st.code(code_knn_3, language="python")
    st.write("แบ่งข้อมูลออกเป็นชุด Train (80%) และ Test (20%)")
    
    st.write("### 📌 4. ทำการ Scaling ข้อมูล")
    code_knn_4 = """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    """
    st.code(code_knn_4, language="python")
    st.write("ใช้ StandardScaler เพื่อปรับขนาดข้อมูลให้อยู่ในช่วงเดียวกัน")
    
    st.write("### 📌 5. สร้างและฝึกโมเดล KNN")
    code_knn_5 = """
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)
    """
    st.code(code_knn_5, language="python")
    st.write("สร้างโมเดล KNN และทำการ Train โมเดลด้วยข้อมูลที่เตรียมไว้")
    
    st.write("### 📌 6. ทดสอบโมเดล")
    code_knn_6 = """
    accuracy = model.score(X_test_scaled, y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    """
    st.code(code_knn_6, language="python")
    st.write("ทดสอบโมเดลโดยใช้ชุดข้อมูล Test และคำนวณค่าความแม่นยำ")

st.write("---")
st.write("📌 **Datasetจาก : www.kaggle.com**")