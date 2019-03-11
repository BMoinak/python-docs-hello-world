def knnmodel():
    return 20
from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    x = knnmodel()
    y="<h1>Accuracy of the model is</h1>" + str(x)
    return y
