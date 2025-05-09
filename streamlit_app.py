
import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import tensorflow as tf

# عنوان الصفحة
st.set_page_config(page_title="🌍 EuroSAT Classifier", layout="centered")

# تحميل النموذج
@st.cache_resource
def load_eurosat_model():
    model = load_model("eurosat_cnn_model.keras")
    return model

model = load_eurosat_model()

# خريطة التصنيفات (حسب تدريبك - عدل حسب الحاجة)
class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
               'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
               'River', 'SeaLake']

# عنوان الموقع
st.title("🛰️ EuroSAT Land Use Classifier")
st.markdown("Upload a satellite image and the model will classify the land use category.")

# رفع صورة
uploaded_file = st.file_uploader("Choose an image (RGB, 64x64 pixels recommended)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # معالجة الصورة
    image = image.resize((64, 64))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # تنبؤ
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # عرض النتيجة
    st.success(f"🧠 Predicted Class: **{predicted_class}**")
    st.info(f"🔍 Confidence: **{confidence:.2f}%**")

    # عرض احتمالات كل الفئات
    st.subheader("🔢 Prediction Probabilities:")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {prediction[0][i]:.4f}")