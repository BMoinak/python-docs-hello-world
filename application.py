import numpy as np
import pandas as pd
def knnmodel():
    return 20
from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    x = knnmodel()
    y="Accuracy of the model is: " + str(x)
    return y
