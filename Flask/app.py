from flask import Flask, render_template, redirect, url_for
from predictor import predict

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict_route():
    result = predict()
    return render_template('result.html', result=result)

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
