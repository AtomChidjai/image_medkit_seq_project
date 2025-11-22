import streamlit as st
import cv2
import time
from sequencer import Sequencer
from streamlit_sortables import sort_items

st.set_page_config(page_title="MedOrder", layout="wide")
st.title("🚑 MedOrder: Intelligent Object Sequencing for Medical Supplies")
st.divider()

MODEL_PATH = 'model/medx_mini.pt'
SOURCE_INPUT = 0
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7
DEFAULT_IMGSZ = 256
DEFAULT_TOLERANCE = 10
ALERT_DURATION_SECONDS = 3.0

## initialize session state
if "running" not in st.session_state:
    st.session_state.running = False
if "fps" not in st.session_state:
    st.session_state.fps = 0
if "tracking_list" not in st.session_state:
    st.session_state.tracking_list = ['Eno', 'Mybacin', 'Paracetamol']
if "last_log" not in st.session_state:
    st.session_state.last_log = None
if "log_filename" not in st.session_state:
    st.session_state.log_filename = None

# list of sequences
PRESET_SEQUENCES = {
    "Pharmaceuticals": ['Eno', 'Mybacin', 'Paracetamol'],
    "Simple Meds": ['Eno', 'Mybacin'],
    "Long Run": ['Mybacin', 'Paracetamol', 'Eno', 'Paracetamol'],
}

# log download and stop
def handle_stop_and_log(manual_reset: bool = False):
    if 'sequencer' in st.session_state and st.session_state.running:
        # Log the stop event
        if manual_reset:
            st.session_state.sequencer.logger.info("--- MANUAL RESET ---")
        else:
            st.session_state.sequencer.logger.info("--- RUN STOPPED BY USER ---")
        
        # Extract log data
        st.session_state.last_log = st.session_state.sequencer.get_log_content()
        st.session_state.log_filename = f"log_{st.session_state.sequencer.session_id}.txt"
        
        # Request the main loop to break
        st.session_state.running = False
    elif st.session_state.running:
        st.session_state.running = False


def handle_manual_reset():
    if 'sequencer' in st.session_state and st.session_state.running:
        handle_stop_and_log(manual_reset=True)
        st.session_state.sequencer.reset_state()
        st.warning("⚠️ Sequencer stopped and reset. Press Start to resume.")


def update_alert_display(alert_placeholder, sequencer):
    state_info = sequencer.get_state_info()
    
    # Show error alert during VALIDATING state
    if state_info['is_validating']:
        # Extract the wrong item message
        if ':' in sequencer.message:
            error_detail = sequencer.message.split(':', 1)[1].strip()
        else:
            error_detail = sequencer.message
        
        alert_placeholder.error(f"🚨 **WRONG ITEM!** {error_detail}", icon="❌")
    else:
        alert_placeholder.empty()


def format_status_message(sequencer) -> str:
    state_info = sequencer.get_state_info()
    state = state_info['state']
    progress = state_info['progress']
    active_count = state_info['active_objects']
    
    # State emoji mapping
    state_emoji = {
        'IDLE': '⏸️',
        'TRACKING': '🔍',
        'VALIDATING': '⚠️',
        'WAIT_FOR_CLEAR': '✅'
    }
    
    emoji = state_emoji.get(state, '❓')
    
    # Build status message
    status_parts = [
        f"{emoji} **State:** {state}",
        f"**Progress:** {progress}",
        f"**Active Objects:** {active_count}"
    ]
    
    return " | ".join(status_parts)

# Layout
col1, col2 = st.columns([1, 1])

# right column (control)
with col2:
    st.subheader("⚙️ Model & Control Panel")
    
    alert_placeholder = st.empty()
    
    st.markdown("#### Status")
    status_text = st.empty()
    fps_display = st.empty()
    
    # Sequence Selection
    st.markdown("#### Sequence Selection")
    
    selected_key = st.selectbox(
        "Select a Tracking Sequence:",
        options=list(PRESET_SEQUENCES.keys()),
        key="sequence_key"
    )
    
    # Update tracking list when sequence changes
    if "prev_sequence_key" not in st.session_state:
        st.session_state.prev_sequence_key = selected_key
        st.session_state.tracking_list = PRESET_SEQUENCES[selected_key]
    elif st.session_state.prev_sequence_key != selected_key:
        st.session_state.tracking_list = PRESET_SEQUENCES[selected_key]
        st.session_state.prev_sequence_key = selected_key
    
    # Active Sequence Sorting
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
        st.info("📄 Log file ready for download from the last session.")
    
    # Model Parameters
    st.markdown("#### Model Parameters")
    conf = st.slider("Confidence Threshold", 0.0, 1.0, DEFAULT_CONF, key='conf_slider')
    imgsz = st.slider("Image Size", 128, 640, DEFAULT_IMGSZ, step=32, key='imgsz_slider')
    tolerance = st.slider("Border Tolerance (px)", 1, 20, DEFAULT_TOLERANCE, key='tol_slider')
    iou = st.slider("IoU Threshold", 0.0, 1.0, DEFAULT_IOU, key='iou_slider')

# left column (camera)
with col1:
    st.subheader("📹 Real-Time Camera Feed")
    frame_window = st.empty()
    
    start, stop, reset = st.columns(3)
    
    # Start Button
    with start:
        if st.button("▶️ Start Sequencer", use_container_width=True) and not st.session_state.running:
            # clear log info
            st.session_state.last_log = None
            st.session_state.log_filename = None
            
            # create new sequencer instance
            st.session_state.sequencer = Sequencer(
                model_path=MODEL_PATH,
                tracking_list=st.session_state.tracking_list,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                tolerance=tolerance
            )
            st.session_state.running = True
    
    # Stop Button
    with stop:
        if st.button("⏹️ Stop Sequencer", use_container_width=True, on_click=handle_stop_and_log):
            pass
    
    # Manual Reset Button
    with reset:
        if st.button("🔄 Manual Reset (X)", use_container_width=True, on_click=handle_manual_reset):
            pass

# main loop
if st.session_state.running:
    # init camera
    cap = cv2.VideoCapture(SOURCE_INPUT, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        st.error("❌ Cannot access camera.")
        st.session_state.running = False
    else:
        fps_time = time.time()
        
        # Main processing loop
        while st.session_state.running and cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                status_text.warning("⚠️ Cannot grab frame. Camera stream ended.")
                st.session_state.sequencer.logger.error("Stream capture failed/ended.")
                handle_stop_and_log()
                break
            
            # Process frame through sequencer
            processed_frame, status_message = st.session_state.sequencer.process_frame(frame)
            
            # update alert display
            update_alert_display(alert_placeholder, st.session_state.sequencer)
            
            # convert to RGB for Streamlit display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # display frame
            frame_window.image(processed_frame_rgb, channels="RGB")
            
            # update status with state information
            formatted_status = format_status_message(st.session_state.sequencer)
            status_text.markdown(f"### {formatted_status}")
            
            # calculate and display FPS
            current_time = time.time()
            fps = 1.0 / (current_time - fps_time) if (current_time - fps_time) > 0 else 0
            st.session_state["fps"] = round(fps, 1)
            fps_display.markdown(f"**FPS:** {st.session_state['fps']}")
            fps_time = current_time

            time.sleep(0.01)
        
        # Cleanup after loop exits
        cap.release()
        if 'sequencer' in st.session_state:
            del st.session_state.sequencer
else:
    st.info("ℹ️ Click **Start Sequencer** to begin streaming.")