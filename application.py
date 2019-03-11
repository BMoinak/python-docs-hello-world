def knnmodel():
    return 20
from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "<H1>Accuracy of KNN is </H1>"
