import streamlit as st
import os
import tempfile
import numpy as np
from PIL import Image

# Load YOLO/OpenCV safely so the app does not crash with a traceback.
try:
    import cv2
    from ultralytics import YOLO
    IMPORT_ERROR = None
except Exception as e:
    cv2 = None
    YOLO = None
    IMPORT_ERROR = str(e)

st.set_page_config(
    page_title="YOLOv8 Vehicle Segmentation",
    page_icon="🚗",
    layout="wide"
)

st.title("YOLOv8 Vehicle Segmentation App")
st.write("Upload an image or video to perform vehicle segmentation.")

if IMPORT_ERROR is not None:
    st.error("YOLO/OpenCV could not be loaded.")
    st.warning(
        "The Streamlit environment is likely using an incompatible "
        "Python/OpenCV combination. See the error below."
    )
    st.code(IMPORT_ERROR)
    st.stop()


@st.cache_resource
def load_model():
    model_path = "yolov8n-seg.pt"

    if not os.path.exists(model_path):
        st.error(
            "yolov8n-seg.pt was not found. Put this model file "
            "in the same GitHub folder as app.py."
        )
        st.stop()

    return YOLO(model_path)


model = load_model()

uploaded_file = st.file_uploader(
    "Choose an image or video file...",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file is not None:

    file_type = uploaded_file.type or ""

    # IMAGE
    if file_type.startswith("image/"):
        st.subheader("Processing Image...")

        try:
            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)

            st.image(
                image,
                caption="Original Image",
                use_container_width=True
            )

            with st.spinner("Running YOLOv8 segmentation..."):
                results = model(
                    img_array,
                    conf=0.5,
                    iou=0.7,
                    verbose=False
                )

            segmented_bgr = results[0].plot()
            segmented_rgb = cv2.cvtColor(
                segmented_bgr,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                segmented_rgb,
                caption="Segmented Image",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Could not process the image: {e}")

    # VIDEO
    elif file_type.startswith("video/"):
        st.subheader("Processing Video...")

        tmp_video_path = None
        output_video_path = None

        try:
            suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_video_path = tmp.name

            cap = cv2.VideoCapture(tmp_video_path)

            if not cap.isOpened():
                st.error("Could not open the uploaded video.")
                st.stop()

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                fps = 25.0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            output_video_path = os.path.join(
                tempfile.gettempdir(),
                "segmented_output.mp4"
            )

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            out = cv2.VideoWriter(
                output_video_path,
                fourcc,
                fps,
                (frame_width, frame_height)
            )

            if not out.isOpened():
                st.error("Could not create the output video.")
                cap.release()
                st.stop()

            progress = st.progress(0)
            status = st.empty()
            frame_count = 0

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                results = model(
                    frame,
                    conf=0.5,
                    iou=0.7,
                    verbose=False
                )

                segmented_frame = results[0].plot()
                out.write(segmented_frame)

                frame_count += 1

                if total_frames > 0:
                    progress.progress(
                        min(frame_count / total_frames, 1.0)
                    )

                status.write(
                    f"Processing frame {frame_count}"
                    + (
                        f" / {total_frames}"
                        if total_frames > 0
                        else ""
                    )
                )

            cap.release()
            out.release()

            progress.progress(1.0)
            status.empty()

            st.success("Video processing complete!")
            st.video(output_video_path)

            with open(output_video_path, "rb") as f:
                st.download_button(
                    "Download Segmented Video",
                    data=f.read(),
                    file_name="segmented_output.mp4",
                    mime="video/mp4"
                )

        except Exception as e:
            st.error(f"Could not process the video: {e}")

        finally:
            if tmp_video_path and os.path.exists(tmp_video_path):
                try:
                    os.remove(tmp_video_path)
                except Exception:
                    pass

    else:
        st.warning("Please upload a JPG, PNG, MP4, AVI, or MOV file.")
