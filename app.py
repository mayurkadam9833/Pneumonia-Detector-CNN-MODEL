import streamlit as st 
import numpy as np 
import joblib
import cv2

# load model
model=joblib.load("model.h5")

# Streamlit UI
st.title("🩺 Pneumonia Detection from Chest X-Ray")
st.write("Upload a chest X-ray image and let the model predict.")
uploaded_file=st.file_uploader(label="upload xray",type=["jpg","jpeg","png"])

# function for predictiongi
def detect(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred < 0.5:
        return "NORMAL"
    else:
        return "PNEUMONIA"
    
if uploaded_file is not None:
    # Save the uploaded image temporarily
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display uploaded image
    st.image("temp.jpg", caption="Uploaded X-ray", use_column_width=True)

    if st.button("Predict"):
        result = detect("temp.jpg")
        st.success(f"Prediction: **{result}**")