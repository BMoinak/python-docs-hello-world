import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D,MaxPooling1D, Flatten
from keras import backend as K
from keras.layers import LSTM, GRU, SimpleRNN
from keras.callbacks import CSVLogger
from sklearn.ensemble import VotingClassifier
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import h5py
import pickle

data = pd.read_csv("test.csv")
X = data.drop("fraud", axis = 1)
y = data["fraud"]
y_num = y.values
def cnnmodel():
    test_labels = np.array(y)
    test_features = np.reshape(X.values, (X.values.shape[0],X.values.shape[1],1))
    lstm_output_size = 70
    cnn = Sequential()
    cnn.add(Convolution1D(64, 3, activation="relu", input_shape= (8, 1)))
    cnn.add(Convolution1D(64, 3, activation="relu"))
    cnn.add(MaxPooling1D(pool_size=2))
    cnn.add(Convolution1D(128, 3, activation="relu", padding = "same"))
    cnn.add(Convolution1D(128, 3, activation="relu", padding = "same"))
    cnn.add(MaxPooling1D(pool_size=2))
    cnn.add(LSTM(lstm_output_size))
    cnn.add(Dropout(0.1))
    cnn.add(Dense(2, activation="softmax"))
    cnn.load_weights("cnn_model.hdf5")
    cnn.compile(loss="sparse_categorical_crossentropy", optimizer="SGD", metrics=['accuracy'])
    print("Created model and loaded weights from file")
    y_cnn = cnn.predict_classes(test_features)
    return accuracy_score(y_num, y_cnn)
def knnmodtrain():
    with open("knn.pkl", "rb") as f:
        knn = pickle.load(f)
    y_knn = knn.predict(X)
    return accuracy_score(y_num,y_knn)
from flask import Flask, redirect, url_for, request
app = Flask(__name__)

@app.route("/poststuff",methods = ['POST','GET'])
def hello():
    #knnmodel()
    y=""
    if request.method == 'POST':
        user = request.form['nm']
        y=user+"<br>"
    return y

@app.route("/")
def hello():
    #knnmodel()
    x = knnmodtrain()
    z = cnnmodel()
    y="Accuracy of the KNN model is: " + str(x) + "<br>\n Accuracy of the CNN Model is :" + str(z)
    return y
