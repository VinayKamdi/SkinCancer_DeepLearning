import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

# Load both models
binary_model = load_model('model.keras')  # Binary classification (Benign/Malignant)
multiclass_model = load_model('c_model.h5')  # Multiclass classification (Skin diseases)

# Define class names for multiclass classification
multiclass_class_names = ['Actinic Keratoses', 'Basal Cell Carcinoma', 'Benign Keratosis-like Lesions', 'Dermatofibroma', 'Melanoma', 'Melanocytic Nevi', 'Vascular Lesions']
binary_class_names = ['Benign', 'Malignant']

st.title("Skin Cancer Prediction System")

uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)

    # Preprocess image for both models
    def preprocess_image(uploaded_image, target_size):
        resized_image = uploaded_image.resize(target_size)
        image_array = img_to_array(resized_image)
        image_array /= 255.0  # Normalize
        return image_array

    # Prediction function for binary classification model
    def predict_binary(image_array):
        pred = binary_model.predict(np.expand_dims(image_array, axis=0))
        pred_class = np.argmax(pred)
        confidence = pred[0][pred_class]
        pred_class_name = binary_class_names[pred_class]
        return pred_class_name, confidence

    # Prediction function for multiclass classification model
    def predict_multiclass(image_array):
        pred = multiclass_model.predict(np.expand_dims(image_array, axis=0))
        pred_class = np.argmax(pred)
        confidence = pred[0][pred_class]
        pred_class_name = multiclass_class_names[pred_class]
        return pred_class_name, confidence

    # Preprocess the image for binary model (256x256) and multiclass model (180x180)
    binary_preprocessed_image = preprocess_image(image, (256, 256))
    multiclass_preprocessed_image = preprocess_image(image, (180, 180))

    # Predict using both models
    binary_pred_class, binary_confidence = predict_binary(binary_preprocessed_image)
    multiclass_pred_class, multiclass_confidence = predict_multiclass(multiclass_preprocessed_image)

    # Display results for binary classification in tabular form
    binary_result_df = pd.DataFrame({
        "Class": [binary_pred_class],
        "Confidence (%)": [round(binary_confidence * 100, 2)]
    })
    st.table(binary_result_df)

    # Display results for multiclass classification in tabular form
    multiclass_result_df = pd.DataFrame({
        "Class": [multiclass_pred_class],
        "Confidence (%)": [round(multiclass_confidence * 100, 2)]
    })
    st.table(multiclass_result_df)

    # Provide additional information for better interpretation
    st.markdown("### Interpretation Guide")
    st.markdown("""
    - **Benign:** The lesion is non-cancerous.
    - **Malignant:** The lesion is cancerous and requires immediate medical attention.
    - **Actinic Keratoses:** Precancerous spots caused by sun damage.
    - **Basal Cell Carcinoma:** A common form of skin cancer that is usually treatable.
    - **Benign Keratosis-like Lesions:** Non-cancerous skin growths.
    - **Dermatofibroma:** A benign skin nodule.
    - **Melanoma:** A serious form of skin cancer that requires prompt treatment.
    - **Melanocytic Nevi:** Commonly known as moles, usually benign.
    - **Vascular Lesions:** Abnormalities in the blood vessels of the skin.
    """)
