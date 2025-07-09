from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/professional", methods=["GET", "POST"])
def professional():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            input_data = pd.DataFrame([[
                float(request.form["acousticness"]),
                float(request.form["danceability"]),
                float(request.form["energy"]),
                float(request.form["instrumentalness"]),
                float(request.form["liveness"]),
                float(request.form["loudness"]),
                float(request.form["speechiness"]),
                float(request.form["tempo"]),
                float(request.form["valence"]),
                int(request.form["duration_ms"]),
                request.form["genre"],
                request.form["key"],
                request.form["mode"],
                request.form["time_signature"]
            ]], columns=[
                "acousticness", "danceability", "energy", "instrumentalness", "liveness",
                "loudness", "speechiness", "tempo", "valence", "duration_ms",
                "genre", "key", "mode", "time_signature"
            ])
            prediction = model.predict(input_data)[0]
        except Exception as e:
            error = str(e)
    return render_template("pro.html", prediction=prediction, error=error)

@app.route("/common", methods=["GET", "POST"])
def common():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            input_data = pd.DataFrame([[
                float(request.form["acousticness"]),
                float(request.form["danceability"]),
                float(request.form["energy"]),
                request.form["genre"]
            ]], columns=[
                "acousticness", "danceability", "energy", "genre"
            ])
            prediction = model.predict(input_data)[0]
        except Exception as e:
            error = str(e)
    return render_template("common.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)