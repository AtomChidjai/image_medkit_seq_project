# app.py

import streamlit as st
import cv2
import time
import os
import io 
# Import the Sequencer class from the separate module
from sequencer import Sequencer 
import numpy as np # Re-added numpy import
from streamlit_sortables import sort_items

st.set_page_config(page_title="YOLO Object Detection", layout="wide")

st.title("🚑 MedOrder: Intelligent Object Sequencing for Medical Supplies")
st.divider()

# --- State Initialization ---
if "running" not in st.session_state:
    st.session_state.running = False
if "fps" not in st.session_state:
    st.session_state.fps = 0
if "sequencer" not in st.session_state:
    st.session_state.tracking_list = ['Eno', 'Mybacin', 'Paracetamol'] 
if "last_log" not in st.session_state:
    st.session_state.last_log = None
if "log_filename" not in st.session_state:
    st.session_state.log_filename = None

# --- Global Configs ---
MODEL_PATH = 'medx_mini.pt'
SOURCE_INPUT = 0 
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7
DEFAULT_IMGSZ = 256
DEFAULT_TOLERANCE = 10

col1, col2 = st.columns([1, 1])

# --- DOWNLOAD LOG FUNCTION (Callback) ---
def handle_stop_and_log(manual_reset=False):
    """Handles stopping the stream and extracting log data."""
    if 'sequencer' in st.session_state and st.session_state.running:
        # 1. Log the stop event
        if manual_reset:
            st.session_state.sequencer.logger.info("--- MANUAL RESET (X) PRESSED ---")
        else:
            st.session_state.sequencer.logger.info("--- RUN STOPPED BY USER ---")
        
        # 2. Extract log data
        st.session_state.last_log = st.session_state.sequencer.get_log_content()
        st.session_state.log_filename = f"log_{st.session_state.sequencer.session_id}.txt"
        
        # 3. Request the main loop to break
        st.session_state.running = False
        
        # We don't delete sequencer here, we wait for the loop to fully exit.
    elif st.session_state.running:
         st.session_state.running = False

# --- MANUAL RESET HANDLER ---
def handle_manual_reset():
    """Triggers the stop handler, then performs the soft reset."""
    if 'sequencer' in st.session_state and st.session_state.running:
        handle_stop_and_log(manual_reset=True) # This sets running=False and prepares the log
        # Ensure reset happens on the next run if user starts immediately
        st.session_state.sequencer.reset_state()
        st.warning("Sequencer stopped and reset requested. Press Start to resume.")


with col2:
    st.subheader("⚙️ Model & Control Panel")
    
    alert_placeholder = st.empty() 

    st.markdown("#### Status")
    status_text = st.empty() 
    
    # ... (Sequence Selection and Sorting Logic remains the same) ...
    PRESET_SEQUENCES = {
    "Pharmaceuticals": ['Eno', 'Mybacin', 'Paracetamol'],
    "Simple Meds": ['Eno', 'Mybacin'], 
    "Long Run": ['Mybacin', 'Paracetamol', 'Eno', 'Paracetamol'],
    }

    st.markdown("#### Sequence Selection")

    selected_key = st.selectbox(
        "Select a Tracking Sequence:",
        options=list(PRESET_SEQUENCES.keys()),
        key="sequence_key"
    )

    if selected_key not in st.session_state:
        st.session_state.tracking_list = PRESET_SEQUENCES[selected_key]
    elif st.session_state.sequence_key != selected_key:
        st.session_state.tracking_list = PRESET_SEQUENCES[selected_key]
        st.session_state.sequence_key = selected_key


    st.markdown("#### Active Sequence Sorting")
    st.caption("Drag and drop to adjust the order for the currently selected sequence.")

    sorted_items = sort_items(st.session_state.tracking_list) 
    st.session_state.tracking_list = sorted_items

    st.write(sorted_items)
    
    st.markdown("---")

    # Log Download Section
    if st.session_state.last_log and not st.session_state.running:
        log_bytes = st.session_state.last_log.encode('utf-8')
        st.download_button(
            label=f"⬇️ Download Log File: {st.session_state.log_filename}",
            data=log_bytes,
            file_name=st.session_state.log_filename,
            mime="text/plain",
            key="download_log_button"
        )
        # st.session_state.last_log = None # Don't clear here, only on start
        # st.session_state.log_filename = None
        st.info("Log file ready for download from the last session.")

    st.markdown("#### Model Parameters")
    conf = st.slider("Confidence Threshold", 0.0, 1.0, DEFAULT_CONF, key='conf_slider')
    imgsz = st.slider("Image Size", 128, 640, DEFAULT_IMGSZ, step=32, key='imgsz_slider')
    tolerance = st.slider("Border Tolerance (px)", 1, 20, DEFAULT_TOLERANCE, key='tol_slider')
    iou = st.slider("IoU Threshold", 0.0, 1.0, DEFAULT_IOU, key='iou_slider')

with col1:
    st.subheader("Live Camera Feed")
    frame_window = st.empty()

    start, stop, reset = st.columns(3)
    
    # --- START BUTTON LOGIC ---
    with start:
        if st.button("Start Sequencer", use_container_width=True) and not st.session_state.running:
            # Clear previous log info upon starting a new session
            st.session_state.last_log = None
            st.session_state.log_filename = None
            
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
        if st.button("Stop Sequencer", use_container_width=True, on_click=handle_stop_and_log):
             # This button click immediately executes handle_stop_and_log
             pass
    
    # --- MANUAL RESET BUTTON LOGIC ---
    with reset:
        if st.button("Manual Reset (X)", use_container_width=True, on_click=handle_manual_reset):
            # This button click immediately executes handle_manual_reset
            pass

    # --- MAIN LOOP ---
    if st.session_state.running:
        cap = cv2.VideoCapture(SOURCE_INPUT, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            st.error("Cannot access camera. Check device index or DSHOW backend.")
            st.session_state.running = False
            # No break here, rely on the conditional running check below
        
        fps_time = time.time()

        while st.session_state.running and cap.isOpened(): # Loop termination relies solely on this condition
            ret, frame = cap.read()
            if not ret:
                status_text.warning("Cannot grab frame. Camera stream ended.")
                st.session_state.sequencer.logger.error("Stream capture failed/ended.")
                handle_stop_and_log()
                break # Hard break on capture failure
            
            # Use Sequencer to process the frame
            processed_frame, status_message = st.session_state.sequencer.process_frame(frame)
            
            # --- ALERT SYSTEM ---
            if st.session_state.sequencer.last_event_type == 'Wrong':
                alert_placeholder.error(f"🚨 **WRONG ITEM!** {st.session_state.sequencer.message.split(':')[1].strip()}", icon="❌")
            elif st.session_state.sequencer.last_event_type == 'Correct':
                 alert_placeholder.empty()
            else:
                 alert_placeholder.empty()
            
            # Convert to RGB for Streamlit display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame and status
            frame_window.image(processed_frame_rgb, channels="RGB")
            status_text.markdown(f"## **Current Status:** {status_message}")
            
            # FPS calculation and update
            current_time = time.time()
            fps = 1.0 / (current_time - fps_time) if (current_time - fps_time) > 0 else 0
            st.session_state["fps"] = round(fps, 1)
            fps_time = current_time
            
            time.sleep(0.01)

        # Cleanup: This block runs immediately after the loop condition (st.session_state.running) becomes False
        cap.release()
        del st.session_state.sequencer # Explicitly delete the object after the loop is done
        
    else:
        st.info("Click **Start Sequencer** to begin streaming.")