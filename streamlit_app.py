
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

# ==========================================
# STAGE 3: MACHINE LEARNING
# ==========================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

st.divider()
st.header("🧠 Stage 3: Machine Learning")

st.warning(
    "SIMULATION ONLY: This model learns from "
    "artificially generated vibration trials. "
    "It has not been validated on a physical bridge."
)

# Generate independent simulated trials

def create_ml_dataset():

    rng = np.random.default_rng(2026)
    rows = []

    settings = {
        "Baseline": (5.0, 0.10),
        "Added Mass": (3.5, 0.13),
        "Connection Variation": (2.5, 0.18)
    }

    sample_rate = 50
    duration = 5
    time = np.arange(0, duration, 1 / sample_rate)

    for label, (base_frequency, base_amplitude) in settings.items():

        for trial_id in range(60):

            frequency = base_frequency + rng.normal(0, 0.25)
            amplitude = base_amplitude * rng.uniform(0.8, 1.2)

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

            features = analyse_vibration(
                x, y, z, sample_rate
            )

            features["Condition"] = label
            features["Trial ID"] = f"{label}_{trial_id}"

            rows.append(features)

    return pd.DataFrame(rows)


# Build dataset

ml_data = create_ml_dataset()

feature_columns = [
    "RMS",
    "Standard Deviation",
    "Peak-to-Peak",
    "Dominant Frequency (Hz)"
]

X = ml_data[feature_columns]
y = ml_data["Condition"]

# Split complete trials into training and testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Train Random Forest

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Simple baseline for comparison

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

model_accuracy = accuracy_score(y_test, predictions)
baseline_accuracy = accuracy_score(
    y_test, baseline_predictions
)

# Display dataset

st.subheader("📚 Simulated Training Dataset")

st.write("Total independent trials:", len(ml_data))
st.write("Training trials:", len(X_train))
st.write("Testing trials:", len(X_test))

st.dataframe(
    ml_data.head(10),
    use_container_width=True,
    hide_index=True
)

# Display model performance

st.subheader("📊 Model Evaluation")

col1, col2 = st.columns(2)

col1.metric(
    "Random Forest Accuracy",
    f"{model_accuracy * 100:.2f}%"
)

col2.metric(
    "Simple Baseline Accuracy",
    f"{baseline_accuracy * 100:.2f}%"
)

st.caption(
    "These scores measure performance on held-out simulated trials, "
    "not real bridge performance."
)

# Confusion matrix

st.subheader("🔍 Confusion Matrix")

class_names = list(settings.keys()) if False else [
    "Baseline",
    "Added Mass",
    "Connection Variation"
]

cm = confusion_matrix(
    y_test,
    predictions,
    labels=class_names
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

st.write("Rows = actual condition; columns = predicted condition.")

st.dataframe(
    cm_df,
    use_container_width=True
)

# ------------------------------------------
# STAGE 4.2 - EXPERIMENT HISTORY STORAGE
# ------------------------------------------

if "experiment_history" not in st.session_state:
    st.session_state.experiment_history = []
# ------------------------------------------
# AI PREDICTION WITH CONFIDENCE ANALYSIS
# ------------------------------------------

st.subheader("🔮 Current Trial AI Classification")

# Prepare current trial features

current_features = pd.DataFrame(
    [[metrics[name] for name in feature_columns]],
    columns=feature_columns
)

# Predict bridge condition

current_prediction = model.predict(current_features)[0]

# Calculate class probabilities

probabilities = model.predict_proba(current_features)[0]

# Match probabilities with their condition names

probability_df = pd.DataFrame({
    "Condition": model.classes_,
    "Probability": probabilities
})

probability_df["Percentage"] = (
    probability_df["Probability"] * 100
)

# Display predicted condition

st.markdown("### 🧠 AI Prediction")

st.success(
    f"Predicted Bridge Condition: {current_prediction}"
)

# Display model confidence

confidence = np.max(probabilities) * 100

st.metric(
    "Model Confidence",
    f"{confidence:.2f}%"
)

# Display all condition probabilities

st.markdown("### 📊 Condition Probability Analysis")

for _, row in probability_df.iterrows():

    st.write(
        f"**{row['Condition']} — "
        f"{row['Percentage']:.2f}%**"
    )

    st.progress(
        float(row["Probability"])
    )

# Compare selected condition with prediction

st.markdown("### 🔍 Experiment Comparison")

col1, col2 = st.columns(2)

col1.metric(
    "Selected Condition",
    condition
)

col2.metric(
    "AI Predicted Condition",
    current_prediction
)

if condition == current_prediction:

    st.success(
        "The AI prediction matches the selected "
        "simulated experiment condition."
    )

else:

    st.warning(
        "The AI prediction differs from the selected "
        "simulated experiment condition."
    )

st.caption(
    "IMPORTANT: This model uses simulated data. "
    "Confidence percentages are model probability estimates, "
    "not verified real-world reliability or structural safety."
)
# ------------------------------------------
# SAVE EXPERIMENT
# ------------------------------------------

st.divider()
st.header("🗂️ Stage 4.2: Experiment History")

if st.button("💾 Save Experiment to History"):

    already_saved = any(
        item["Trial"] == int(trial)
        and item["Selected Condition"] == condition
        for item in st.session_state.experiment_history
    )

    if already_saved:

        st.warning(
            "This trial and condition have already been saved."
        )

    else:

        experiment = {
            "Trial": int(trial),
            "Selected Condition": condition,
            "AI Prediction": current_prediction,
            "AI Probability (%)": round(float(confidence), 2),
            "RMS": round(float(metrics["RMS"]), 4),
            "Dominant Frequency (Hz)": round(
                float(metrics["Dominant Frequency (Hz)"]), 4
            )
        }

        st.session_state.experiment_history.append(experiment)

        st.success("Experiment saved successfully!")

# ------------------------------------------
# DISPLAY EXPERIMENT HISTORY
# ------------------------------------------

st.subheader("📋 Previous Experiments")

if st.session_state.experiment_history:

    history_df = pd.DataFrame(
        st.session_state.experiment_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    st.write(
        "Total saved experiments:",
        len(history_df)
    )

else:

    st.info(
        "No experiments saved yet. "
        "Click Save Experiment to record your first trial."
    )
# ------------------------------------------
# STAGE 4.3 - MULTIPLE EXPERIMENT COMPARISON
# ------------------------------------------

st.divider()
st.header("📊 Stage 4.3: Experiment Comparison")

if len(st.session_state.experiment_history) >= 2:

    comparison_history = pd.DataFrame(
        st.session_state.experiment_history
    )

    st.subheader("RMS Comparison")

    st.bar_chart(
        comparison_history.set_index("Trial")["RMS"]
    )

    st.subheader("Dominant Frequency Comparison")

    st.bar_chart(
        comparison_history.set_index("Trial")[
            "Dominant Frequency (Hz)"
        ]
    )

    st.subheader("AI Prediction Comparison")

    st.dataframe(
        comparison_history[
            [
                "Trial",
                "Selected Condition",
                "AI Prediction",
                "AI Probability (%)"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Save at least two experiments to compare them."
    )

# Feature importance

st.subheader("📌 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

st.bar_chart(
    importance_df.set_index("Feature")
)
