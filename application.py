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
dev knnmodel():
    full_data=pd.read_csv("train.csv")
    full_data=full_data.sample(frac=1)
    full_labels = full_data['fraud']
    full_features = full_data.drop('fraud', axis = 1)
    full_features_array = full_features.values
    full_labels_array = full_labels.values
    train_features,test_features,train_labels,test_labels=train_test_split(
    full_features_array,full_labels_array,train_size=0.80,test_size=0.20)
    train_features=normalize(train_features)
    test_features=normalize(test_features)
    knn=KNeighborsClassifier(n_neighbors=4,algorithm="kd_tree",n_jobs=-1)
    knn.fit(train_features,train_labels.ravel())
    knn_predicted_test_labels=knn.predict(test_features)
    tn,fp,fn,tp=confusion_matrix(test_labels,knn_predicted_test_labels).ravel()
    knn_accuracy_score=accuracy_score(test_labels,knn_predicted_test_labels)
    knn_precison_score=precision_score(test_labels,knn_predicted_test_labels)
    knn_recall_score=recall_score(test_labels,knn_predicted_test_labels)
    knn_f1_score=f1_score(test_labels,knn_predicted_test_labels)
    return knn_accuracy_score
from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return ("<H1>Accuracy of KNN is </H1>"+knnmodel())
