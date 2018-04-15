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

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score




def gen_feature_matrix(song_score_dict,label_dict):

    feature_matrix = np.zeros((len(song_score_dict),5001))
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


def svm_main(song_score_dict,test_score_dict,label_dict,test_label_dict):
    

    #kernels = ['linear', 'poly','rbf' ]

    X_train,y_train = gen_feature_matrix(song_score_dict,label_dict)

    X_test, y_true = gen_feature_matrix(test_score_dict,test_label_dict)
    
    svm_diff_kernels(X_train,y_train,X_test, y_true)
    #train/test split
    #split_idx = int(len(feature_matrix)*.70)
    #split_idx = int(len(feature_matrix)*1)


    # X_train,y_train = feature_matrix[:split_idx],labels[:split_idx]
    # X_test, y_true = feature_matrix[split_idx:],labels[split_idx:]


    # classifier = OneVsRestClassifier(SVC(kernel='linear'))
    # classifier.fit(X_train,y_train)
    # y_pred = classifier.predict(X_test)
    # print("y_pred: ", y_pred)
    # print("y_true: ", y_true)
    # accuracy = accuracy_score(y_true,y_pred)
    # print("svm accuracy: ",accuracy)


    #calculate accuracy or something

def svm_diff_kernels(X_train,y_train,X_test, y_true):

    # classifier = OneVsRestClassifier(SVC(kernel='linear'))
    # classifier.fit(X_train,y_train)
    # y_pred = classifier.predict(X_test)
    # print("y_pred: ", y_pred)
    # print("y_true: ", y_true)
    # accuracy = accuracy_score(y_true,y_pred)
    # print("linear svm accuracy: ",accuracy, "\n")

    #for precision on songs of 2000s
    #y_pred_2000s =[0 if decade != 2000 else 1 for decade in y_pred ]
    y_true_2000s = [0 if decade != '200' else 1 for decade in y_true ]
    y_train_2000s = [0 if decade != '200' else 1 for decade in y_train ]
    #class_binary = OneVsRestClassifier(SVC(kernel='linear'))
    class_binary = SVC(kernel='linear')
    # print(y_train_2000s)
    # print("space\n")
    # print(X_train)
    # time.sleep(20)
    class_binary.fit(X_train,y_train_2000s)
    
    #y_pred_2000s = class_binary.predict(X_test)
    #accuracy = accuracy_score(y_true_2000s,y_pred_2000s)
    #print("linear binary svm for 2000s accuracy: ", accuracy)
    
    y_scores = class_binary.decision_function(X_test)
    precision, recall, thresholds = precision_recall_curve(y_true_2000s, y_scores)
    precision, recall, _ = precision_recall_curve(y_true_2000s, y_scores)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    average_precision = average_precision_score(y_true_2000s, y_scores)

    print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Two-class Precision-Recall curve for Songs from 2000s: AP={0:0.2f}'.format(
          average_precision))
    plt.show()
    plt.savefig('Precision_Recall.png')

    # classifier = OneVsRestClassifier(SVC(kernel='rbf'))
    # classifier.fit(X_train,y_train)
    # y_pred = classifier.predict(X_test)
    # print("y_pred: ", y_pred)
    # print("y_true: ", y_true)
    # accuracy = accuracy_score(y_true,y_pred)
    # print("rbf svm accuracy: ",accuracy, "\n")

    # classifier = OneVsRestClassifier(SVC(kernel='poly'))
    # classifier.fit(X_train,y_train)
    # y_pred = classifier.predict(X_test)
    # print("y_pred: ", y_pred)
    # print("y_true: ", y_true)
    # accuracy = accuracy_score(y_true,y_pred)
    # print("polynomial svm accuracy: ",accuracy, "\n")

















