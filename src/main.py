
# ============================================================
# UrbanSense-AI
# AI-Powered Urban Environment Analysis
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------
# Create output directory
# ------------------------------------------------------------

os.makedirs("data", exist_ok=True)


# ------------------------------------------------------------
# Advanced Environmental Analysis
# ------------------------------------------------------------

def analyze_environment(df):

    print("\n" + "=" * 50)
    print("ADVANCED ENVIRONMENTAL ANALYSIS")
    print("=" * 50)

    # Average Temperature
    avg_temp = df["Temperature"].mean()

    # Average AQI
    avg_aqi = df["AQI"].mean()

    # Average Noise Level
    avg_noise = df["Noise_Level"].mean()

    print(f"\nAverage Temperature: {avg_temp:.2f} °C")
    print(f"Average AQI: {avg_aqi:.2f}")
    print(f"Average Noise Level: {avg_noise:.2f} dB")

    # Most polluted city
    most_polluted = df.loc[df["AQI"].idxmax()]

    print("\nMost Polluted City:")
    print(f"City: {most_polluted['City']}")
    print(f"AQI: {most_polluted['AQI']}")

    # Hottest city
    hottest_city = df.loc[df["Temperature"].idxmax()]

    print("\nHottest City:")
    print(f"City: {hottest_city['City']}")
    print(f"Temperature: {hottest_city['Temperature']} °C")

    # Noisiest city
    noisiest_city = df.loc[df["Noise_Level"].idxmax()]

    print("\nNoisiest City:")
    print(f"City: {noisiest_city['City']}")
    print(f"Noise Level: {noisiest_city['Noise_Level']} dB")

    # Environmental classification
    df["Air_Quality_Status"] = np.where(
        df["AQI"] <= 100,
        "Moderate",
        "Poor"
    )

    print("\nCity-wise Environmental Status:")
    print(df[["City", "Air_Quality_Status"]])


# ------------------------------------------------------------
# Main Function
# ------------------------------------------------------------

def main():

    print("=" * 50)
    print("WELCOME TO URBANSENSE-AI")
    print("AI-Powered Urban Environment Analysis")
    print("=" * 50)

    # --------------------------------------------------------
    # Create Environmental Dataset
    # --------------------------------------------------------

    data = {
        "City": [
            "Lucknow",
            "Delhi",
            "Mumbai",
            "Kolkata",
            "Chennai"
        ],

        "Temperature": [
            32,
            38,
            31,
            34,
            33
        ],

        "AQI": [
            120,
            180,
            95,
            150,
            110
        ],

        "Noise_Level": [
            65,
            78,
            70,
            75,
            68
        ]
    }

    df = pd.DataFrame(data)

    # --------------------------------------------------------
    # Display Dataset
    # --------------------------------------------------------

    print("\nEnvironmental Dataset:")
    print(df)

    # --------------------------------------------------------
    # Dataset Summary
    # --------------------------------------------------------

    print("\nDataset Summary:")
    print(df.describe())

    # --------------------------------------------------------
    # Advanced Environmental Analysis
    # --------------------------------------------------------

    analyze_environment(df)

    # --------------------------------------------------------
    # Data Visualization
    # --------------------------------------------------------

    print("\nGenerating Environmental Graphs...")

    sns.set_theme(style="whitegrid")

    # --------------------------------------------------------
    # Temperature Graph
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    sns.barplot(
        x="City",
        y="Temperature",
        data=df
    )

    plt.title("City-wise Temperature Analysis")
    plt.xlabel("City")
    plt.ylabel("Temperature (°C)")

    plt.tight_layout()
    plt.savefig("data/temperature.png")

    plt.show()

    # --------------------------------------------------------
    # AQI Graph
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    sns.barplot(
        x="City",
        y="AQI",
        data=df
    )

    plt.title("City-wise Air Quality Index Analysis")
    plt.xlabel("City")
    plt.ylabel("AQI")

    plt.tight_layout()
    plt.savefig("data/aqi.png")

    plt.show()

    # --------------------------------------------------------
    # Noise Level Graph
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    sns.barplot(
        x="City",
        y="Noise_Level",
        data=df
    )

    plt.title("City-wise Noise Pollution Analysis")
    plt.xlabel("City")
    plt.ylabel("Noise Level (dB)")

    plt.tight_layout()
    plt.savefig("data/noise.png")

    plt.show()

    # --------------------------------------------------------
    # Correlation Heatmap
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    correlation = df[
        ["Temperature", "AQI", "Noise_Level"]
    ].corr()

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Environmental Parameter Correlation")

    plt.tight_layout()
    plt.savefig("data/correlation.png")

    plt.show()

    # --------------------------------------------------------
    # Save Dataset
    # --------------------------------------------------------

    df.to_csv(
        "data/environmental_analysis.csv",
        index=False
    )

    print("\nEnvironmental dataset saved successfully!")

    print("\nGenerated Files:")
    print("1. data/temperature.png")
    print("2. data/aqi.png")
    print("3. data/noise.png")
    print("4. data/correlation.png")
    print("5. data/environmental_analysis.csv")

    print("\n" + "=" * 50)
    print("URBANSENSE-AI PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 50)


# ------------------------------------------------------------
# Program Entry Point
# ------------------------------------------------------------

if __name__ == "__main__":
    main()
