
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Load environmental dataset
df = pd.read_csv("data/environmental_analysis.csv")

# Select input features
X = df[["Temperature", "Noise_Level"]]

# Target variable
y = df["AQI"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("===== UrbanSense-AI ML Model =====")
print("Model: Random Forest")
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# Predict AQI for a new city
new_city = pd.DataFrame({
    "Temperature": [35],
    "Noise_Level": [72]
})

predicted_aqi = model.predict(new_city)

print("\nNew City Prediction")
print("Temperature: 35 °C")
print("Noise Level: 72 dB")
print("Predicted AQI:", round(predicted_aqi[0], 2))
