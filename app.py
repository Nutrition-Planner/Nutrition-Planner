from flask import Flask, request, jsonify, send_from_directory
import pickle
import pandas as pd
from flask_cors import CORS
import os

app = Flask(
    __name__,
    static_folder="../frontend",
    template_folder="../frontend"
)

CORS(app)

# Load ML models
rf_model = pickle.load(open("rf_model.pkl", "rb"))
le_goal = pickle.load(open("le_goal.pkl", "rb"))
le_pref = pickle.load(open("le_pref.pkl", "rb"))
le_city = pickle.load(open("le_city.pkl", "rb"))


# SERVE FRONTEND FILES

@app.route("/")
def serve_index():
    return send_from_directory(app.template_folder, "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)


# PREDICT CALORIES ENDPOINT

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    Age = float(data["Age"])
    Weight = float(data["Weight"])
    Height = float(data["Height"])
    ActivityFactor = float(data["ActivityFactor"])
    Preference = data["Preference"]
    Goal = data["Goal"]
    BMR = float(data["BMR"])
    city = data["city"].title()
    Temperature = float(data["Temperature"])
    Humidity = float(data["Humidity"])

    diet_map = {"Veg": 0, "Vegan": 1, "Non-Veg": 2, "Pescatarian": 3}
    health_map = {"Maintain": 0, "Gain": 1, "Loss": 2}
    city_map = {
        "Mumbai": 0, "Delhi": 1, "Pune": 2, "Hyderabad": 3,
        "Chennai": 4, "Bengaluru": 5, "Kolkata": 6, "Jaipur": 7,
        "Surat": 8, "Lucknow": 9
    }

    input_df = pd.DataFrame([{
        "Age": Age,
        "Weight": Weight,
        "Height": Height,
        "ActivityFactor": ActivityFactor,
        "Preference": diet_map[Preference],
        "Goal": health_map[Goal],
        "BMR": BMR,
        "city": city_map[city],
        "Temperature": Temperature,
        "Humidity": Humidity
    }])

    prediction = float(rf_model.predict(input_df)[0])
    return jsonify({"PredictedCalories": prediction})


# MEAL RECOMMENDER ENDPOINT

@app.route("/meal_recommend", methods=["POST"])
def meal_recommend():
    data = request.json
    calorie_target = int(data.get("calorie_target", 2000))
    preference = data.get("preference", "any")
    goal = data.get("goal", "maintain")
    city = data.get("city", "")
    temp = float(data.get("temp", 25))

    from meal_recommender import MealRecommender
    mr = MealRecommender(food_csv="food1_cleaned.csv", users_csv=None)
    plan = mr.recommend_meal_plan(
        calorie_target=calorie_target,
        preference=preference,
        goal=goal,
        temp=temp
    )
    return jsonify(plan)


# START SERVER

if __name__ == "__main__":
    app.run(debug=True, port=5000)
