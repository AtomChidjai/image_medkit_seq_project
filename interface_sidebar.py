import streamlit as st
import cv2
from camera import Camera
import time

# --- Page Setup ---
st.set_page_config(page_title="YOLO Object Detection", layout="wide")

# --- Sidebar (Right side panel) ---
with st.sidebar:
    st.title("⚙️ Control Panel")

    st.markdown("### 🔧 Model Parameters")
    conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)
    imgsz = st.slider("Image Size", 128, 640, 256, step=32)
    st.markdown("---")

    st.markdown("### 🧩 Controls")
    start_process = st.button("▶ Start Process", use_container_width=True)
    stop_process = st.button("⏹ Stop Process", use_container_width=True)
    if start_process:
        st.success("Process started ✅")
    if stop_process:
        st.warning("Process stopped ⛔")

    st.markdown("---")

    st.markdown("### ⚠️ Alerts")
    st.info("ℹ️ System running normally")
    st.success("✅ Model loaded successfully")
    st.warning("⚠️ Low light may affect accuracy")
    st.error("❌ No error detected")
    st.markdown("---")

    st.markdown("### 🎚️ Adjust Model Parameters")
    slider1 = st.slider("Detection Sensitivity", 0, 100, 50)
    slider2 = st.slider("IoU Threshold", 0.0, 1.0, 0.5)
    st.markdown("---")

    st.markdown("### 📊 Model Metrics")
    acc_col, fps_col, loss_col = st.columns(3)
    with acc_col:
        st.metric("Accuracy", "95%")
    with fps_col:
        st.metric("FPS", st.session_state.get("fps", 0))
    with loss_col:
        st.metric("Loss", "0.05")

# --- Main Content Area (Camera) ---
st.title("🎯 Real-time Object Detection Dashboard")

if "running" not in st.session_state:
    st.session_state.running = False

col1, _ = st.columns([2, 1])  # left: camera, right sidebar already exists

with col1:
    st.subheader("📷 Live Camera Feed")
    frame_window = st.image([], use_container_width=True)

    start, stop = st.columns(2)
    with start:
        if st.button("▶ Start Camera", use_container_width=True):
            st.session_state.running = True
    with stop:
        if st.button("⏹ Stop Camera", use_container_width=True):
            st.session_state.running = False

    if st.session_state.running:
        yolo_cam = Camera(model_path="multimed_2.pt", conf=conf, imgsz=imgsz)
        cap = cv2.VideoCapture(0)
        fps_time = time.time()

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Cannot access camera.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = yolo_cam.process(frame)
            frame_window.image(processed, channels="RGB")

            # Calculate FPS
            fps = 1.0 / (time.time() - fps_time)
            st.session_state["fps"] = round(fps, 1)
            fps_time = time.time()

            time.sleep(0.03)  # prevent Streamlit overload

        cap.release()
        st.session_state.running = False
    else:
        st.info("Click ▶ **Start Camera** to begin streaming.")
