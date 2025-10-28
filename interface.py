import streamlit as st

st.set_page_config(page_title="Object Detection Interface", layout="wide")

# --- Sidebar ---
st.sidebar.title("Model Controls")
st.sidebar.markdown("### Model Parameters")
model_param_1 = st.sidebar.slider("Parameter 1", 0.0, 1.0, 0.5)
model_param_2 = st.sidebar.slider("Parameter 2", 0, 100, 50)
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Metrics")
st.sidebar.text("Accuracy: 0.95")
st.sidebar.text("FPS: 15")
st.sidebar.text("Loss: 0.05")

# --- Main Interface ---
st.title("Object Detection Interface")

# Camera Section
st.markdown("### Camera Window")
camera_col1, camera_col2 = st.columns([3, 1])
with camera_col1:
    st.text("Camera feed will appear here")
with camera_col2:
    if st.button("Open Camera"):
        st.success("Camera Opened")
    if st.button("Close Camera"):
        st.warning("Camera Closed")

st.markdown("---")

# Buttons Section
st.markdown("### Controls")
button_col1, button_col2, button_col3 = st.columns(3)
with button_col1:
    st.text("Sortable Buttons (Sequence Re-arranging)")
    st.text("Button A | Button B | Button C")  # Placeholder
with button_col2:
    if st.button("Start"):
        st.success("Started")
with button_col3:
    if st.button("Stop"):
        st.warning("Stopped")

st.markdown("---")

# Alerts
st.markdown("### Alert Interface")
st.info("Information Alert")
st.success("Success Alert")
st.warning("Warning Alert")
st.error("Error Alert")

st.markdown("---")

# Model Parameter Section
st.markdown("### Model Parameters Interface")
param1_col, param2_col = st.columns(2)
with param1_col:
    param1 = st.number_input("Parameter 1", min_value=0.0, max_value=1.0, value=0.5)
with param2_col:
    param2 = st.number_input("Parameter 2", min_value=0, max_value=100, value=50)

st.markdown("---")

# Slider Parameters
st.markdown("### Slider Model Parameters")
slider_param1 = st.slider("Slider 1", 0, 100, 50)
slider_param2 = st.slider("Slider 2", 0.0, 1.0, 0.5)

st.markdown("---")

# Model Metrics Interface
st.markdown("### Model Metrics Interface")
metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    st.metric(label="Accuracy", value="0.95")
with metric_col2:
    st.metric(label="FPS", value="15")
with metric_col3:
    st.metric(label="Loss", value="0.05")