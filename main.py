import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# Load the YOLOv8 model
model = YOLO("best.pt")

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        
        # Predict
        results = self.model(image)
        image = self.draw_boxes(image, results)
        
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def draw_boxes(self, image, results):
        image_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.load_default()
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = box.cls[0]
                label = self.model.names[int(cls)]
                # Draw rectangle
                draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
                # Draw label
                text = f"{label} {conf:.2f}"
                text_size = draw.textbbox((0, 0), text, font=font)
                draw.rectangle([x1, y1 - text_size[3], x1 + text_size[2], y1], fill="green")
                draw.text((x1, y1 - text_size[3]), text, fill="white", font=font)
        return np.array(image_pil)

def run_webcam_detection():
    st.title("YOLOv8 Object Detection with Webcam")
    st.write("Click the button below to start the webcam.")

    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

run_webcam_detection()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Predict
    results = model(np.array(image))
    image_np = np.array(image)
    
    # Draw boxes using Pillow
    image_pil = Image.fromarray(image_np)
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.load_default()
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = box.cls[0]
            label = model.names[int(cls)]
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            # Draw label
            text = f"{label} {conf:.2f}"
            text_size = draw.textbbox((0, 0), text, font=font)
            draw.rectangle([x1, y1 - text_size[3], x1 + text_size[2], y1], fill="green")
            draw.text((x1, y1 - text_size[3]), text, fill="white", font=font)
    image_np = np.array(image_pil)
    
    # Display the output image with bounding boxes
    st.image(image_np, caption='Detected Image.', use_column_width=True)
