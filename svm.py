import os,sys
import numpy as np
import os
from os.path import abspath
# from tokenizer import *
# import Stemmer
import time
import copy
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time




def gen_feature_matrix(song_score_dict,label_dict):

    feature_matrix = np.zeros((len(song_score_dict),5000))
    song_names = [name for name in song_score_dict]
    labels = [None]*len(song_names)
    
    i = 0
    for song in song_names:
        labels[i] = label_dict[song]
        i +=1
    print(labels)
    #time.sleep(10)

    for i in range(len(song_names)):

        # if a word isn't in a song, the score will stay zero
        for tup in song_score_dict[song_names[i]]:
            #feature_matrix[i][tup[0]] = song_score_dict[song_names[i]]
            #print(tup[0])
            feature_matrix[i][tup[0]] = tup[1]

    return feature_matrix, labels


def svm_main(song_score_dict,label_dict):
    feature_matrix,labels = gen_feature_matrix(song_score_dict,label_dict)
    #train/test split
    split_idx = int(len(feature_matrix)*.70)
    #split_idx = int(len(feature_matrix)*1)


    X_train,y_train = feature_matrix[:split_idx],labels[:split_idx]
    X_test, y_true = feature_matrix[split_idx:],labels[split_idx:]

    classifier = OneVsRestClassifier(SVC(kernel='linear'))
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    print(y_pred)
    print(y_true)
    accuracy = accuracy_score(y_true,y_pred)
    print(accuracy)


    #calculate accuracy or something















