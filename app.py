from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("hotel_cancellation_model.pkl")
model_columns = joblib.load("model_columns.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        data = {
            "lead_time": int(request.form["lead_time"]),
            "adults": int(request.form["adults"]),
            "children": int(request.form["children"]),
            "babies": int(request.form["babies"]),
            "stays_in_weekend_nights": int(request.form["weekend"]),
            "stays_in_week_nights": int(request.form["weekdays"]),
            "previous_cancellations": int(request.form["previous_cancellations"]),
            "booking_changes": int(request.form["booking_changes"]),
            "total_of_special_requests": int(request.form["special_requests"]),
            "is_repeated_guest": int(request.form["repeated_guest"]),
            "required_car_parking_spaces": int(request.form["parking"]),
            "days_in_waiting_list": int(request.form["waiting"]),
            "adr": float(request.form["adr"]),
            "arrival_date_year": 2017,
            "arrival_date_month": "January",
            "arrival_date_week_number": 1,
            "arrival_date_day_of_month": 1,
            "meal": "BB",
            "country": "Unknown",
            "market_segment": "Online TA",
            "distribution_channel": "TA/TO",
            "reserved_room_type": "A",
            "assigned_room_type": "A",
            "deposit_type": "No Deposit",
            "customer_type": "Transient",
            "hotel": request.form["hotel"],
            "is_repeated_guest": int(request.form["repeated_guest"]),
            "previous_bookings_not_canceled": 0,
            "agent": 0,
            "company": 0,
        }

        data["children"] = float(data["children"])

        input_df = pd.DataFrame([data])
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        result = model.predict(input_df)[0]

        prediction = (
            "Booking is likely to be cancelled."
            if result == 1
            else "Booking is likely to be kept."
        )

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)