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
def knnmodtrain( a_knn, b_knn, c_knn, d_knn, e_knn, f_knn, g_knn, h_knn ):
    with open("knn.pkl", "rb") as f:
        knn = pickle.load(f)
    arr_testing=pd.DataFrame({'customer':[a_knn],'age':[b_knn],'gender':[c_knn],'zipcodeOri':[d_knn],'merchant':[e_knn],'zipMerchant':[f_knn],'category':[g_knn],'amount':[h_knn]},columns=['customer','age','gender','zipcodeOri','merchant','zipMerchant','category','amount'])
    y_knn = knn.predict(arr_testing)
    return y_knn
from flask import Flask, redirect, url_for, request
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hel():
    #knnmodel()
    x = knnmodtrain()
    z = cnnmodel()
    y="Accuracy of the KNN model is: " + str(x) + "<br>\n Accuracy of the CNN Model is :" + str(z)
    return y

@app.route('/poststuff/', methods=['GET', 'POST'])
def hello():
    #knnmodel()
    if request.method == 'POST':
        a_knn = request.form['customer']
        b_knn = request.form['age']
        c_knn = request.form['gender']
        d_knn = request.form['zipcodeOri']
        e_knn = request.form['merchant']
        f_knn = request.form['zipMerchant']
        g_knn = request.form['category']
        h_knn = request.form['amount']
        y=knnmodtrain( a_knn, b_knn, c_knn, d_knn, e_knn, f_knn, g_knn, h_knn )
        return y
    return -1

