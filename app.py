import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from detect_app import detect_license_plates
import tempfile
import io
from streamlit_keycloak import login

keycloak = login(
    url="http://",  # Replace with your Keycloak server URL
    realm="ttt",  # Replace with your Keycloak realm
    client_id="ttt-client"  # Replace with your Keycloak client ID
)

def process_video(video_file):
    vidcap = cv2.VideoCapture(video_file)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []
    detected_texts = []
    frame_nmr = -1
    while True:
        frame_nmr += 1
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_nmr * fps)
        success, image = vidcap.read()
        if not success:
            break
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # For Streamlit display
        frame, texts = detect_license_plates(frame.copy(), frame_nmr)  # Use .copy() to avoid modifying the original frame
        frames.append(frame)
        detected_texts.extend(texts)  # Accumulate all detected texts
    vidcap.release()
    return frames, detected_texts

if keycloak.authenticated:
    st.title("License Plate Detection")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_nmr = -1
        placeholder = st.empty()  # Create an empty placeholder for the table
        data_result = []
        while cap.isOpened():
            frame_nmr += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nmr * fps)
            success, frame = cap.read()
            if not success:
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            img = Image.open(io.BytesIO(frame)).convert('RGB')
            open_cv_image = np.array(img)
            open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
            frame, texts = detect_license_plates(open_cv_image.copy(), frame_nmr)  # Use .copy() to avoid modifying the original frame
            if frame is not None:
                with col1:
                    FRAME_WINDOW.image(frame)
                if len(texts) > 0:
                    data_result.append([frame_nmr + 1, texts])
                    if data_result:  # Check if data_result is not empty
                        df = pd.DataFrame(data_result, columns=['Frame Number', 'License Number'])  # Assign column names here
                        placeholder.table(df)  # Update the table in each iteration
        cap.release()
    #cv2.destroyAllWindows()

