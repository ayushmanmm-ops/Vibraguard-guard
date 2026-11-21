
import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="VIBRAGUARD",
    page_icon="🌉",
    layout="wide"
)

# -------------------------------
# PROJECT HEADER
# -------------------------------

st.title("🌉 VIBRAGUARD")
st.subheader("AI-Assisted Structural Vibration Monitoring")

st.warning(
    "SIMULATION MODE — No physical sensor is connected. "
    "All readings below are artificially generated."
)

# -------------------------------
# SIMULATION SETTINGS
# -------------------------------

st.sidebar.header("CONTROL PANEL")

conditions = [
    "Baseline",
    "Added Mass",
    "Connection Variation"
]

condition = st.sidebar.selectbox(
    "Select bridge condition",
    conditions
)

trial = st.sidebar.number_input(
    "Trial Number",
    min_value=1,
    max_value=1000,
    value=1,
    step=1
)

# -------------------------------
# GENERATE SIMULATED DATA
# -------------------------------

def generate_signal(selected_condition, seed=42):

    rng = np.random.default_rng(seed)

    sample_rate = 50
    duration = 5

    time = np.arange(
        0,
        duration,
        1 / sample_rate
    )

    settings = {
        "Baseline": (5.0, 0.10),
        "Added Mass": (3.5, 0.13),
        "Connection Variation": (2.5, 0.18)
    }

    frequency, amplitude = settings[selected_condition]

    x = (
        amplitude * np.sin(2 * np.pi * frequency * time)
        + rng.normal(0, 0.025, len(time))
    )

    y = (
        amplitude * 0.7 * np.sin(
            2 * np.pi * (frequency + 1) * time
        )
        + rng.normal(0, 0.025, len(time))
    )

    z = (
        amplitude * 0.5 * np.sin(
            2 * np.pi * frequency * time + 0.5
        )
        + rng.normal(0, 0.025, len(time))
    )

    return time, x, y, z, sample_rate


# -------------------------------
# VIBRATION ANALYSIS
# -------------------------------

def analyse_vibration(x, y, z, sample_rate):

    magnitude = np.sqrt(x**2 + y**2 + z**2)

    rms = np.sqrt(np.mean(magnitude**2))

    standard_deviation = np.std(magnitude)

    peak_to_peak = (
        np.max(magnitude) - np.min(magnitude)
    )

    frequencies = np.fft.rfftfreq(
        len(magnitude),
        d=1 / sample_rate
    )

    fft_values = np.abs(
        np.fft.rfft(magnitude - np.mean(magnitude))
    )

    if len(fft_values) > 1:
        dominant_frequency = frequencies[
            1 + np.argmax(fft_values[1:])
        ]
    else:
        dominant_frequency = 0

    return {
        "RMS": rms,
        "Standard Deviation": standard_deviation,
        "Peak-to-Peak": peak_to_peak,
        "Dominant Frequency (Hz)": dominant_frequency
    }


# -------------------------------
# CURRENT TRIAL
# -------------------------------

time, x, y, z, sample_rate = generate_signal(
    condition,
    seed=int(trial)
)

metrics = analyse_vibration(
    x, y, z, sample_rate
)

st.header("📊 Monitoring Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Selected Condition", condition)
col2.metric("Trial Number", int(trial))
col3.metric("Data Source", "Simulated")

# -------------------------------
# VIBRATION GRAPH
# -------------------------------

st.header("📈 Vibration Analysis")

signal_data = pd.DataFrame({
    "Time (seconds)": time,
    "X Axis": x,
    "Y Axis": y,
    "Z Axis": z
})

st.line_chart(
    signal_data.set_index("Time (seconds)")
)

# -------------------------------
# CALCULATED FEATURES
# -------------------------------

st.header("🔬 Vibration Measurements")

c1, c2 = st.columns(2)

c1.metric("RMS", f"{metrics['RMS']:.4f}")
c1.metric(
    "Standard Deviation",
    f"{metrics['Standard Deviation']:.4f}"
)

c2.metric(
    "Peak-to-Peak Amplitude",
    f"{metrics['Peak-to-Peak']:.4f}"
)

c2.metric(
    "Dominant Frequency",
    f"{metrics['Dominant Frequency (Hz)']:.2f} Hz"
)

st.caption(
    "Measurements are calculated from simulated signals. "
    "They are not real bridge measurements."
)

# -------------------------------
# CONDITION COMPARISON
# -------------------------------

st.header("📊 Comparison of Bridge Conditions")

comparison_rows = []

for index, item in enumerate(conditions):

    t, cx, cy, cz, sr = generate_signal(
        item,
        seed=100 + index
    )

    result = analyse_vibration(cx, cy, cz, sr)

    comparison_rows.append({
        "Condition": item,
        **result
    })

comparison_df = pd.DataFrame(comparison_rows)

st.dataframe(
    comparison_df.round(4),
    use_container_width=True,
    hide_index=True
)

st.bar_chart(
    comparison_df.set_index("Condition")[
        ["RMS", "Standard Deviation", "Peak-to-Peak"]
    ]
)

# -------------------------------
# CSV EXPORT
# -------------------------------

st.header("📥 Export Trial Data")

trial_data = signal_data.copy()

trial_data.insert(0, "Condition", condition)
trial_data.insert(1, "Trial", int(trial))
trial_data["Data Source"] = "Simulated"

csv_data = trial_data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Trial Data (CSV)",
    data=csv_data,
    file_name=f"vibraguard_trial_{int(trial)}.csv",
    mime="text/csv"
)

# -------------------------------
# PROJECT INFORMATION
# -------------------------------

st.header("🧪 Experiment Information")

st.write("**Project:** VIBRAGUARD")
st.write("**Planned Sensor:** MPU6050")
st.write("**Planned Controller:** ESP32")
st.write("**Planned AI:** Machine Learning Classification")

st.header("🚀 Development Progress")

st.write("✅ Dashboard interface")
st.write("✅ Simulated vibration signals")
st.write("✅ RMS and standard deviation calculations")
st.write("✅ Peak-to-peak and frequency calculations")
st.write("✅ Condition comparison")
st.write("✅ CSV data export")
st.write("⬜ Real sensor integration")
st.write("⬜ Experimental dataset collection")
st.write("⬜ Machine learning model training")
st.write("⬜ Physical bridge validation")

st.divider()

st.caption(
    "VIBRAGUARD | Engineering Challenge Prototype | Simulation Mode"
)
