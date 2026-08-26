
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import tempfile

# Load a pretrained YOLOv8 segmentation model
# The model file should be in the same directory as app.py or accessible via a path
@st.cache_resource
def load_model():
    model = YOLO('yolov8n-seg.pt')
    return model

model = load_model()

st.title("YOLOv8 Vehicle Segmentation App")
st.write("Upload an image or video to perform vehicle segmentation.")

uploaded_file = st.file_uploader("Choose an image or video file...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None:
    file_type = uploaded_file.type

    if "image" in file_type:
        st.subheader("Processing Image...")
        # Read image
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        # Display original image
        st.image(image, caption="Original Image", use_column_width=True)

        # Run inference
        results = model(img_array, conf=0.5, iou=0.7)

        # Get segmented image
        im_bgr = results[0].plot() # plot() returns BGR image
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

        # Display segmented image
        st.image(im_rgb, caption="Segmented Image", use_column_width=True)

    elif "video" in file_type:
        st.subheader("Processing Video... (This might take a while)")

        # Create a temporary file to save the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split(".")[-1]) as tmp_video_file:
            tmp_video_file.write(uploaded_file.read())
            tmp_video_path = tmp_video_file.name

        cap = cv2.VideoCapture(tmp_video_path)

        if not cap.isOpened():
            st.error(f"Error: Could not open video file {uploaded_file.name}")
        else:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            output_video_path = os.path.join(tempfile.gettempdir(), "segmented_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            progress_bar = st.progress(0)
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Run inference
                results = model(frame, conf=0.5, iou=0.7)

                # Get the segmented image
                im_bgr_segmented = results[0].plot()
                out.write(im_bgr_segmented)

                frame_count += 1
                progress_bar.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            out.release()
            os.remove(tmp_video_path) # Clean up temporary uploaded file

            st.success("Video processing complete!")
            st.video(output_video_path)
            
            with open(output_video_path, "rb") as file:
                st.download_button(
                    label="Download Segmented Video",
                    data=file.read(),
                    file_name="segmented_output.mp4",
                    mime="video/mp4",
                )
            os.remove(output_video_path) # Clean up temporary output file

