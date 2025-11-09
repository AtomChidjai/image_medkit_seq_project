# app.py

import streamlit as st
import cv2
import time
import os
# Import the Sequencer class from the separate module
from sequencer import Sequencer 
import numpy as np
from streamlit_sortables import sort_items

st.set_page_config(page_title="YOLO Object Detection", layout="wide")

st.title("Object Detection Sequencer")
st.divider()

# --- State Initialization ---
if "running" not in st.session_state:
    st.session_state.running = False
if "fps" not in st.session_state:
    st.session_state.fps = 0
if "sequencer" not in st.session_state:
    # Default list for display
    st.session_state.tracking_list = ['Eno', 'Mybacin', 'Paracetamol'] 
    
# --- Global Configs ---
MODEL_PATH = 'medx_mini.pt'
SOURCE_INPUT = 0 
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7
DEFAULT_IMGSZ = 256
DEFAULT_TOLERANCE = 10

col1, col2 = st.columns([1, 1])

with col2:
    st.subheader("⚙️ Model & Control Panel")

    st.markdown("#### Sequence Sorting")
    # Use st.session_state.tracking_list as the default and update it after sorting
    # Removed the 'entity_types' argument to fix the TypeError.
    sorted_items = sort_items(st.session_state.tracking_list) 
    st.session_state.tracking_list = sorted_items
    
    st.markdown("#### Model Parameters")
    conf = st.slider("Confidence Threshold", 0.0, 1.0, DEFAULT_CONF, key='conf_slider')
    imgsz = st.slider("Image Size", 128, 640, DEFAULT_IMGSZ, step=32, key='imgsz_slider')
    tolerance = st.slider("Border Tolerance (px)", 1, 20, DEFAULT_TOLERANCE, key='tol_slider')
    iou = st.slider("IoU Threshold", 0.0, 1.0, DEFAULT_IOU, key='iou_slider')
    
    st.markdown("---")
    st.markdown("#### Status")
    status_text = st.empty() # Placeholder for sequence status
    
    st.markdown("#### Model Metrics")
    fps_col, acc_col = st.columns(2)
    with fps_col:
        st.metric("FPS", st.session_state.get("fps", 0))
    with acc_col:
        st.metric("Expected Next Item", st.session_state.tracking_list[0] if st.session_state.tracking_list and st.session_state.running else "N/A")

with col1:
    st.subheader("Live Camera Feed")
    frame_window = st.empty()

    start, stop, reset = st.columns(3)
    
    # --- START BUTTON LOGIC ---
    with start:
        if st.button("Start Sequencer", use_container_width=True) and not st.session_state.running:
            # Initialize Sequencer when starting
            st.session_state.sequencer = Sequencer(
                model_path=MODEL_PATH,
                tracking_list=st.session_state.tracking_list,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                tolerance=tolerance
            )
            st.session_state.running = True
    
    # --- STOP BUTTON LOGIC ---
    with stop:
        if st.button("Stop Sequencer", use_container_width=True) and st.session_state.running:
            st.session_state.running = False
    
    # --- MANUAL RESET BUTTON LOGIC ---
    with reset:
        if st.button("Manual Reset (X)", use_container_width=True) and st.session_state.running:
            if 'sequencer' in st.session_state:
                st.session_state.sequencer.reset_state()

    # --- MAIN LOOP ---
    if st.session_state.running:
        cap = cv2.VideoCapture(SOURCE_INPUT, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            st.error("Cannot access camera. Check device index or DSHOW backend.")
            st.session_state.running = False
        
        fps_time = time.time()

        while st.session_state.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                status_text.warning("Cannot grab frame. Camera stream ended.")
                st.session_state.running = False
                break
            
            # Use Sequencer to process the frame
            processed_frame, status_message = st.session_state.sequencer.process_frame(frame)
            
            # Convert to RGB for Streamlit display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame and status
            frame_window.image(processed_frame_rgb, channels="RGB")
            status_text.markdown(f"**Current Status:** {status_message}")
            
            # FPS calculation and update
            current_time = time.time()
            fps = 1.0 / (current_time - fps_time) if (current_time - fps_time) > 0 else 0
            st.session_state["fps"] = round(fps, 1)
            fps_time = current_time
            
            # Small sleep to prevent burning CPU too hard
            time.sleep(0.01)

        # Cleanup
        cap.release()
        st.session_state.running = False
    else:
        st.info("Click **Start Sequencer** to begin streaming.")