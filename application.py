import numpy as np
from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "<H1>Accuracy of KNN is </H1>"
