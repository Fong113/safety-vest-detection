import streamlit as st
import torch

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load the YOLOv8 model
model = YOLO("best.pt")

def predict(image):
    # Convert image to numpy array
    image_np = np.array(image)
    # Make predictions
    results = model(image_np)
    return results

def draw_boxes(image, results):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = box.cls[0]
            label = model.names[int(cls)]
            # Draw rectangle
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Draw label
            cv2.putText(image, f'{label} {conf:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

def run_webcam_detection():
    st.title("YOLOv8 Object Detection with Webcam")
    st.write("Click the button below to start or stop the webcam.")

    if 'run' not in st.session_state:
        st.session_state.run = False

    start_button = st.button("Start Webcam")
    stop_button = st.button("Stop Webcam")

    if start_button:
        st.session_state.run = True

    if stop_button:
        st.session_state.run = False

    stframe = st.empty()

    cap = None
    if st.session_state.run:
        cap = cv2.VideoCapture(0)

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.session_state.run = False
            break

        # Predict
        results = predict(frame)
        frame = draw_boxes(frame, results)

        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_column_width=True)

    if cap is not None:
        cap.release()

run_webcam_detection()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Predict
    results = predict(image)
    image_np = np.array(image)
    image_np = draw_boxes(image_np, results)
    
    # Display the output image with bounding boxes
    st.image(image_np, caption='Detected Image.', use_column_width=True)