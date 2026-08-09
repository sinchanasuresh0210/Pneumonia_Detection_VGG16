import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Pneumonia Detection using VGG16",
    page_icon="🩻",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f8fc;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #17365d;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    background-color: #ffffff;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    '<div class="title">🩻 Pneumonia Detection from Chest X-Ray</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning Classification using VGG16 Transfer Learning'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Upload a chest X-ray image. The trained VGG16 model "
    "will classify it as NORMAL or PNEUMONIA."
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "pneumonia_vgg16.keras"
    )

    return model


model = load_model()

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Uploaded X-Ray")

        st.image(
            image,
            caption="Chest X-Ray",
            use_container_width=True
        )

    # ----------------------------------------------
    # PREPROCESS IMAGE
    # ----------------------------------------------

    img = image.resize((224, 224))

    img_array = np.array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = tf.keras.applications.vgg16.preprocess_input(
        img_array
    )

    # ----------------------------------------------
    # MODEL PREDICTION
    # ----------------------------------------------

    prediction = model.predict(
        img_array,
        verbose=0
    )[0][0]

    # ----------------------------------------------
    # CLASSIFICATION
    # ----------------------------------------------

    if prediction >= 0.5:

        result = "PNEUMONIA"

        confidence = prediction * 100

    else:

        result = "NORMAL"

        confidence = (1 - prediction) * 100

    # ----------------------------------------------
    # DISPLAY RESULT
    # ----------------------------------------------

    with col2:

        st.subheader("Prediction Result")

        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        if result == "PNEUMONIA":

            st.error("⚠️ PNEUMONIA DETECTED")

        else:

            st.success("✅ NORMAL")

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(
            int(min(confidence, 100))
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

st.markdown("---")

st.subheader("📌 About the Project")

st.write(
    """
    This project uses the VGG16 convolutional neural network
    with transfer learning to classify chest X-ray images into
    two categories: NORMAL and PNEUMONIA.
    """
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Model",
        "VGG16"
    )

with col2:

    st.metric(
        "Input Size",
        "224 × 224"
    )

with col3:

    st.metric(
        "Test Accuracy",
        "84.94%"
    )

st.warning(
    "⚠️ This application is an educational machine-learning "
    "project and is not a medical diagnostic tool."
)