import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load your trained YOLOv10 model
model = YOLO('best.pt')  # Replace with your model path

# Streamlit app UI
st.title('Brain Tumor Detection Using YOLOv10')
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
submit_clicked = st.button("Submit")
# Create three columns for layout (left, middle, right)
col1, col2, col3 = st.columns([2, 1, 2])

# Upload image in the left column
with col1:
    if uploaded_image is not None:
        # Open and display the image with increased size
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=False, width=400)  # Increased width

# Submit button in the middle column

# Display result in the right column
with col3:
    if uploaded_image is not None and submit_clicked:
        # Convert to OpenCV format for YOLO
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Perform inference
        result = model(img_bgr, imgsz=640, conf=0.25)  # Adjust the size and confidence

        # Get the annotated image
        annotated_img = result[0].plot()
        annotated_img_rgb = annotated_img[:, :, ::-1]  # Convert to RGB format for displaying

        # Display annotated image with increased size
        result_pil = Image.fromarray(annotated_img_rgb)
        st.image(result_pil, caption="Detected Brain Tumor", use_column_width=False, width=400)  # Increased width
