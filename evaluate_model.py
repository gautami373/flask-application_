import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("hotel_bookings.csv")

df = df.drop(columns=[
    "reservation_status",
    "reservation_status_date"
])

df["children"] = df["children"].fillna(0)
df["country"] = df["country"].fillna("Unknown")
df["agent"] = df["agent"].fillna(0)
df["company"] = df["company"].fillna(0)

df = df.drop_duplicates()
df = pd.get_dummies(df, drop_first=True)

X = df.drop("is_canceled", axis=1)
y = df["is_canceled"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = joblib.load("hotel_cancellation_model.pkl")
predictions = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, predictions))

sns.heatmap(
    confusion_matrix(y_test, predictions),
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Hotel Booking Cancellation - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.show()