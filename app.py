import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

st.set_page_config(page_title="Fruit Classifier", page_icon="🍎")

# Load the trained model
model = load_model('')  # Use .h5 or .keras as needed

# Define fruit class names (modify this list according to your model)
class_names = []  

# Function to preprocess image and predict
def predict_image(img):
    img = img.resize((100, 100))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    
    # Get the top 5 predictions
    top_5_idx = np.argsort(prediction[0])[::-1][:5]  # Sort and get top 5
    top_5_probs = prediction[0][top_5_idx] * 100  # Convert to percentage

    return top_5_idx, top_5_probs

# Streamlit UI
st.title("Fruit Detection Model")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Get top 5 predictions and their probabilities
    top_5_idx, top_5_probs = predict_image(img)
    
    # Display the top 5 predictions
    st.write("Top 5 Predictions:")
    for i in range(5):
        st.write(f"{class_names[top_5_idx[i]]}: {top_5_probs[i]:.2f}%")
