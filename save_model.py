
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

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

model = HistGradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "hotel_cancellation_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("Model saved successfully!")
