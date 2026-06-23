# app.py
# Stock Market Trend Analysis

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("dataset/stock_data.csv")

# Convert date
df["Date"] = pd.to_datetime(df["Date"])

# Create day column
df["Day"] = range(len(df))

# Features and target
X = df[["Day"]]
y = df["Close"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Prediction
df["Predicted_Close"] = model.predict(X)

# Plot graph
plt.figure(figsize=(10, 5))

plt.plot(
    df["Date"],
    df["Close"],
    marker="o",
    label="Actual Price"
)

plt.plot(
    df["Date"],
    df["Predicted_Close"],
    linestyle="--",
    label="Predicted Price"
)

plt.title("Stock Market Trend Analysis")
plt.xlabel("Date")
plt.ylabel("Stock Price")

plt.legend()
plt.grid()

# Save output
plt.savefig("screenshots/output_graph.png")

# Show graph
plt.show()

print("\nStock Market Trend Analysis Completed")
print(df[["Date", "Close", "Predicted_Close"]])