
import pandas as pd

df = pd.read_csv("hotel_bookings.csv")

print("Original dataset shape:", df.shape)

print("\nMissing values before cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nDuplicate rows:", df.duplicated().sum())

# Remove columns that could reveal the final cancellation outcome
df = df.drop(
    columns=["reservation_status", "reservation_status_date"]
)

# Handle missing values
df["children"] = df["children"].fillna(0)
df["country"] = df["country"].fillna("Unknown")
df["agent"] = df["agent"].fillna(0)
df["company"] = df["company"].fillna(0)

# Remove duplicate rows
df = df.drop_duplicates()

print("\nDataset shape after cleaning:", df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nData cleaning completed.")

