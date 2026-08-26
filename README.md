
# YOLOv8 Vehicle Segmentation Streamlit App

This repository contains a Streamlit application that uses a pre-trained YOLOv8 segmentation model to detect and segment vehicles in images and videos.

## Setup and Run Locally

1.  Clone this repository:
    ```bash
    git clone <your-repo-url>
    cd Deployment
    ```
2.  Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

## Deployment on Streamlit Cloud

1.  Ensure all files (app.py, requirements.txt, yolov8n-seg.pt) are in the root of your GitHub repository or in a designated subfolder.
2.  Go to Streamlit Cloud (share.streamlit.io) and connect your GitHub account.
3.  Select the repository and the main branch, and specify `app.py` as the main file.
4.  Click "Deploy!"

Enjoy detecting and segmenting vehicles!
