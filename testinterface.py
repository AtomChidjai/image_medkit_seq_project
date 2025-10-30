import cv2
import streamlit as st
from camera import Camera

st.title("🧠 Real-Time YOLO Object Tracking")

run_button = st.button("▶ Start Camera")
stop_button = st.button("⏹ Stop Camera")

frame_window = st.image([])

if "running" not in st.session_state:
    st.session_state.running = False

if run_button:
    st.session_state.running = True
if stop_button:
    st.session_state.running = False

if st.session_state.running:
    yolo_cam = Camera(model_path="multimed_2.pt", conf=0.25)

    cap = cv2.VideoCapture(0)

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Cannot access camera.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = yolo_cam.process(frame)

        frame_window.image(processed_frame, channels="RGB")

        # Streamlit needs small sleep for UI updates
        if not st.session_state.running:
            break

    cap.release()
    st.session_state.running = False
else:
    st.write("Click ▶ **Start Camera** to begin.")
