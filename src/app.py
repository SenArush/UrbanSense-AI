
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="UrbanSense-AI",
    page_icon="🌍",
    layout="wide"
)

# ---------------- CUSTOM STYLE ----------------
st.markdown("""
<style>
.main {
    background-color: #f4f7fb;
}
.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🌍 UrbanSense-AI")
st.subheader("AI-Powered Urban Environmental Analysis")

st.write(
    "Analyze city temperature, Air Quality Index (AQI), "
    "and noise pollution using data analytics and machine learning."
)

# ---------------- LOAD DATA ----------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "environmental_analysis.csv"

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    # Fallback dataset
    return pd.DataFrame({
        "City": ["Lucknow", "Delhi", "Mumbai", "Kolkata", "Chennai"],
        "Temperature": [32, 38, 31, 34, 33],
        "AQI": [120, 180, 95, 150, 110],
        "Noise_Level": [65, 78, 70, 75, 68]
    })

df = load_data()

required_columns = [
    "City", "Temperature", "AQI", "Noise_Level"
]

if not all(col in df.columns for col in required_columns):
    st.error("Dataset is missing required columns.")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard",
        "Environmental Analysis",
        "AI Prediction",
        "Dataset"
    ]
)

# ---------------- DASHBOARD ----------------
if page == "Dashboard":

    st.header("📊 Environmental Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Temperature",
        f"{df['Temperature'].mean():.1f} °C"
    )

    col2.metric(
        "Average AQI",
        f"{df['AQI'].mean():.1f}"
    )

    col3.metric(
        "Average Noise",
        f"{df['Noise_Level'].mean():.1f} dB"
    )

    st.divider()

    st.subheader("City-wise Environmental Overview")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=df,
        x="City",
        y="AQI",
        ax=ax
    )

    ax.set_title("Air Quality Index by City")
    ax.set_ylabel("AQI")
    ax.set_xlabel("City")

    plt.xticks(rotation=30)

    st.pyplot(fig)

    st.subheader("Environmental Dataset")

    st.dataframe(df, use_container_width=True)

# ---------------- ENVIRONMENTAL ANALYSIS ----------------
elif page == "Environmental Analysis":

    st.header("🌱 Environmental Analysis")

    selected_city = st.selectbox(
        "Select a city",
        df["City"].unique()
    )

    city_data = df[df["City"] == selected_city].iloc[0]

    st.write(f"### Environmental Report: {selected_city}")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Temperature",
        f"{city_data['Temperature']} °C"
    )

    col2.metric(
        "AQI",
        f"{city_data['AQI']}"
    )

    col3.metric(
        "Noise Level",
        f"{city_data['Noise_Level']} dB"
    )

    st.subheader("Temperature Comparison")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=df,
        x="City",
        y="Temperature",
        ax=ax
    )

    plt.xticks(rotation=30)

    st.pyplot(fig)

    st.subheader("Noise Pollution Comparison")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=df,
        x="City",
        y="Noise_Level",
        ax=ax
    )

    plt.xticks(rotation=30)

    st.pyplot(fig)

    st.subheader("Correlation Between Environmental Factors")

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        df[["Temperature", "AQI", "Noise_Level"]].corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

# ---------------- AI PREDICTION ----------------
elif page == "AI Prediction":

    st.header("🤖 AI-Powered AQI Prediction")

    st.info(
        "Enter environmental values to predict the estimated AQI."
    )

    col1, col2 = st.columns(2)

    with col1:

        temperature = st.slider(
            "Temperature (°C)",
            min_value=0,
            max_value=50,
            value=35
        )

    with col2:

        noise = st.slider(
            "Noise Level (dB)",
            min_value=20,
            max_value=120,
            value=72
        )

    X = df[["Temperature", "Noise_Level"]]
    y = df["AQI"]

    if len(df) >= 2:

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(X, y)

        if st.button("Predict AQI", type="primary"):

            prediction = model.predict(
                pd.DataFrame({
                    "Temperature": [temperature],
                    "Noise_Level": [noise]
                })
            )[0]

            st.success(
                f"Predicted AQI: {prediction:.2f}"
            )

            if prediction <= 50:
                category = "Good"
            elif prediction <= 100:
                category = "Satisfactory"
            elif prediction <= 200:
                category = "Moderate"
            elif prediction <= 300:
                category = "Poor"
            elif prediction <= 400:
                category = "Very Poor"
            else:
                category = "Severe"

            st.write(f"### Estimated Category: {category}")

            st.warning(
                "This is a demonstration model trained on the "
                "available sample dataset. Predictions are not "
                "official air-quality measurements."
            )

    else:
        st.error("At least two dataset records are required.")

# ---------------- DATASET ----------------
elif page == "Dataset":

    st.header("📁 Environmental Dataset")

    st.write(f"Total Records: {len(df)}")

    st.dataframe(df, use_container_width=True)

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Dataset as CSV",
        data=csv_data,
        file_name="urbansense_environmental_data.csv",
        mime="text/csv"
    )

# ---------------- FOOTER ----------------
st.divider()

st.caption(
    "UrbanSense-AI | Environmental Data Analytics "
    "and Machine Learning Project"
)
