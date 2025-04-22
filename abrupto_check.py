# c:\ai_drift_detection\inspect_data.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/mixed_1010_abrupto.csv")
print("Columns:", df.columns)
print("Head:\n", df.head())

# Plot features to visualize abrupt changes
for col in df.columns[:-1]:  # Exclude label
    plt.plot(df[col], label=col)
plt.title("mixed_1010_abrupto.csv Features")
plt.legend()
plt.show()