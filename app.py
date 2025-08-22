import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load YOLOv8 model
model = YOLO("models/yolov8n-face.pt")  # make sure this file exists

st.set_page_config(page_title="Face Detection App", layout="wide")
st.title("🧑‍💻 Face Detection using YOLOv8")

# Sidebar for mode selection
mode = st.sidebar.radio("Choose Detection Mode:", ["Image Upload", "Webcam"])

# ---------------- IMAGE UPLOAD MODE ----------------
if mode == "Image Upload":
    st.subheader("📸 Detect Faces from Uploaded Image")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        # Read image as numpy array
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Detect Faces"):
            results = model(img_array)

            # Annotated image
            annotated_frame = results[0].plot()

            # Count detected faces
            face_count = len(results[0].boxes)

            # Show results
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                     caption="Detected Faces", use_column_width=True)
            st.success(f"✅ Faces detected: {face_count}")

            # Save and download option
            out_path = "detected_faces.jpg"
            cv2.imwrite(out_path, annotated_frame)
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download Annotated Image", f, file_name="detected_faces.jpg")

# ---------------- WEBCAM MODE ----------------
elif mode == "Webcam":
    st.subheader("🎥 Live Face Detection from Webcam")

    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("⚠️ Could not access webcam.")
            break

        # Run YOLO face detection
        results = model(frame)
        annotated_frame = results[0].plot()

        # Show in Streamlit
        FRAME_WINDOW.image(annotated_frame, channels="BGR")

    camera.release()
