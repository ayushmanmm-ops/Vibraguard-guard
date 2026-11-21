
import streamlit as st
import random
import math
import pandas as pd

st.set_page_config(
    page_title="VIBRAGUARD",
    page_icon="🌉",
    layout="wide"
)

st.title("🌉 VIBRAGUARD")
st.subheader("AI-Assisted Structural Vibration Monitoring")

st.warning("SIMULATION MODE — No physical sensor is connected.")

st.sidebar.title("CONTROL PANEL")

condition = st.sidebar.selectbox(
    "Select bridge condition",
    ["Baseline", "Added Mass", "Connection Variation"]
)

if "trial" not in st.session_state:
    st.session_state.trial = 1

if st.sidebar.button("Generate New Trial"):
    st.session_state.trial += 1

st.sidebar.write("Trial:", st.session_state.trial)

settings = {
    "Baseline": (0.25, 3.0),
    "Added Mass": (0.55, 1.8),
    "Connection Variation": (0.95, 4.5)
}

amplitude, frequency = settings[condition]
random.seed(st.session_state.trial)

time = [i * 0.05 for i in range(200)]

x_axis = [
    amplitude * math.sin(2 * math.pi * frequency * t)
    + random.uniform(-0.08, 0.08)
    for t in time
]

y_axis = [
    amplitude * 0.6 * math.sin(2 * math.pi * frequency * 0.7 * t)
    + random.uniform(-0.06, 0.06)
    for t in time
]

z_axis = [
    amplitude * 0.8 * math.sin(2 * math.pi * frequency * 1.2 * t)
    + random.uniform(-0.07, 0.07)
    for t in time
]

st.markdown("## 📊 Monitoring Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Selected Condition", condition)
col2.metric("Trial Number", st.session_state.trial)
col3.metric("Data Source", "Simulated")

st.divider()

st.markdown("## 📈 Vibration Analysis")

chart_data = pd.DataFrame({
    "X Axis": x_axis,
    "Y Axis": y_axis,
    "Z Axis": z_axis
})

st.line_chart(chart_data)

st.divider()

st.markdown("## 🧠 Condition Analysis")

if condition == "Baseline":
    st.success("Baseline category selected.")
elif condition == "Added Mass":
    st.info("Added mass category selected.")
else:
    st.warning("Connection variation category selected.")

st.caption(
    "This is a simulated demonstration, not an AI prediction."
)

st.divider()

st.markdown("## 🧪 Experiment Information")

st.write("Project: VIBRAGUARD")
st.write("Planned sensor: MPU6050")
st.write("Planned controller: ESP32")
st.write("Planned AI: Machine Learning classification")

st.markdown("## 🔬 Development Progress")

st.write("✅ Dashboard interface")
st.write("✅ Simulated vibration signals")
st.write("⬜ Real sensor integration")
st.write("⬜ Experimental dataset collection")
st.write("⬜ Machine learning model training")
st.write("⬜ Physical bridge validation")

st.caption("VIBRAGUARD | Engineering Challenge Prototype")
