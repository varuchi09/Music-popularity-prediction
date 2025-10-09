from flask import Flask, render_template, request
from main import predict_main, predict_general

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/index", methods=["GET"])
def index():
    user_type = request.args.get("type")
    if user_type == "professional":
        return render_template("professional.html")
    elif user_type == "common":
        return render_template("common.html")
    else:
        return "Unknown type", 400
    
@app.route("/predict_pro", methods=["POST"])
def predict_pro():
    form = request.form.to_dict()
    result = predict_main(
        form['genre'],
        form['key'],
        form['mode'],
        form['time_signature'],
        float(form['acousticness']),
        float(form['danceability']),
        int(form['duration_ms']),
        float(form['energy']),
        float(form['instrumentalness']),
        float(form['liveness']),
        float(form['loudness']),
        float(form['speechiness']),
        float(form['tempo']),
        float(form['valence'])
    )
    return render_template("result.html", prediction = result, user = 'Professional')
    
@app.route("/predict_common", methods=["POST"])
def predict_common():
    form = request.form.to_dict()
    result = predict_general(
        form['genre'],
        form['mood'],
        form['tempo_category'],
        form['duration_category']
    )
    return render_template("result.html", prediction = result, user = 'Common User')

if __name__ == "__main__":
    app.run(debug=True)