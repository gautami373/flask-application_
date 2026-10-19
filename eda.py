
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("hotel_bookings.csv")

# 1. Booking cancellation count
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="is_canceled")
plt.title("Booking Cancellation Count")
plt.xlabel("Cancellation (0 = No, 1 = Yes)")
plt.ylabel("Number of Bookings")
plt.savefig("eda1_cancellation_count.png")
plt.show()

# 2. Cancellations by hotel type
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="hotel", hue="is_canceled")
plt.title("Cancellations by Hotel Type")
plt.xlabel("Hotel Type")
plt.ylabel("Number of Bookings")
plt.savefig("eda2_hotel_type.png")
plt.show()

# 3. Lead time and cancellation
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="is_canceled", y="lead_time")
plt.title("Lead Time vs Cancellation")
plt.xlabel("Cancellation (0 = No, 1 = Yes)")
plt.ylabel("Lead Time (Days)")
plt.savefig("eda3_lead_time.png")
plt.show()

# 4. Special requests and cancellation
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="total_of_special_requests",
              hue="is_canceled")
plt.title("Special Requests vs Cancellation")
plt.xlabel("Number of Special Requests")
plt.ylabel("Number of Bookings")
plt.savefig("eda4_special_requests.png")
plt.show()

print("EDA completed. Four charts saved successfully.")
