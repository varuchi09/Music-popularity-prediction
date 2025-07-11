from flask import Flask, render_template, request
from model import predict_popularity, interactive_predict_popularity

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/index", methods=["GET"])
def index():
    user_type = request.args.get("type")
    if user_type == "pro":
        return render_template("pro.html")
    elif user_type == "common":
        return render_template("common.html")
    else:
        return "Unknown type", 400
    
@app.route("/predict_pro", methods=["POST"])
def predict_pro():
    form = request.form.to_dict()
    result = predict_popularity(**form)
    return f"Predicted Popularity Score (Pro): {result}"

@app.route("/predict_common", methods=["POST"])
def predict_common():
    form = request.form.to_dict()
    result = interactive_predict_popularity(form)
    return f"Predicted Popularity Score (Common): {result}"

if __name__ == "__main__":
    app.run(debug=True)