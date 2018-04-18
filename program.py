import os,sys
import numpy as np
import os
from os.path import abspath
# from tokenizer import *
# import Stemmer
import time
import copy
import math
import svm
import matplotlib.pyplot as plt

from collections import Counter

#Unused now, but used to create plots
def plot_NN(X_train, X_test, label_dict):
    k_values = [1,3,5,7,9,15,25,50,75,100]
    accuracies = []
    #Data from previous runs
    total_accuracies = [0.404, 0.454, 0.488, 0.532, 0.546, 0.562, 0.566, 0.558, 0.556, 0.556] #[0.404, 0.488, 0.546]
    accuracies_2000s = [0.579136690647482, 0.6798561151079137, 0.7805755395683454, 0.8705035971223022, 0.8992805755395683, 0.960431654676259, 0.9892086330935251, 1.0, 1.0, 1.0]
    accuracies_1990s = [0.25735294117647056, 0.2426470588235294, 0.18382352941176472, 0.14705882352941177, 0.16176470588235295, 0.10294117647058823, 0.058823529411764705, 0.007352941176470588, 0.0, 0.0]

    for k in k_values:
        accuracies.append(train_NN(song_tfidf_dict, test_dict, label_dict, k))
    print(accuracies)

    plt.step(k_values, accuracies, color='r', where='post')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 100.0])
    plt.title('K vs accuracy for KNN')
    plt.show()
    plt.savefig('knn.png')

def train_NN(X_train, X_test, label_dict, k):
    correct = 0.
    total = 0.
    for query_dict in X_test:
        solution = nearestNeighbor(X_test[query_dict], X_train, label_dict, k)
        if(solution[0] == label_dict[query_dict]):
            correct += 1
        total += 1
    print("KNN accuracy for k = " + str(k) + " is " + str(correct/total))
    return(correct/total)

def cosineDiff(query_dict, year_dict, song_dict, df_dict, N):
    years_tfidf = dict.fromkeys(set(year_dict.values()))
    years_count = dict.fromkeys(set(year_dict.values()))

    for year in years_tfidf:
        years_tfidf[year] = []

    for year in years_count:
        years_count[year] = 0

    for song in song_dict:
        for tup in song_dict[song]:
            year = year_dict[song]
            idx = []
            if years_tfidf[year]:
                idx = [x for x, y in enumerate(years_tfidf[year]) if y[0] == tup[0]]
            if not idx:
                years_tfidf[year].append(tup)
            else:
                years_tfidf[year][idx[0]][1] += tup[1]
            years_count[year] += 1

    for year in years_tfidf:
        if years_count[year] > 0:
            for tup in years_tfidf[year]:
                tup[1] = tup[1] / years_count[year]

    for tup in query_dict:
        tup[1] = tfidf(tup[1],df_dict[tup[0]],N,'tfidf')

    cosines = dict.fromkeys(set(year_dict.values()))
    cosines = dict.fromkeys(cosines, 0)
    querySum = sum([i[1] ** 2 for i in query_dict])

    for year in cosines:
        yearSum = sum([i[1] ** 2 for i in years_tfidf[year]])
        dotProd = 0

        for i in query_dict:
            idx = [x for x, y in enumerate(years_tfidf[year]) if y[0] == i[0]]
            if idx:
                dotProd += i[1] * years_tfidf[year][idx[0]][1]

        if yearSum:
            cosines[year] = dotProd / (math.sqrt(querySum * yearSum))

    return max(cosines, key=cosines.get)


def nearestNeighbor(query_dict, song_dict, label_dict, k):
    tfidf_vals = {}
    for song in song_dict:
        curVal = 0
        counter = 0
        for tup in song_dict[song]:
            while(query_dict[counter][0] < tup[0]):
                counter += 1
                if(counter >= len(query_dict)):
                    break
            if(counter >= len(query_dict)):
                break
            if(query_dict[counter][0] == tup[0]):
                curVal += query_dict[counter][1] * tup[1]
        tfidf_vals[song] = curVal

    best_vals = Counter(tfidf_vals).most_common(k)

    new_counter_dict = {}
    curVal = 2000
    for val in best_vals:
        #print(val[0])
        #exit(1)
        if(label_dict[val[0]] in new_counter_dict):
            new_counter_dict[label_dict[val[0]]] += curVal
        else:
            new_counter_dict[label_dict[val[0]]] = curVal
        curVal -=1

    #print(Counter(new_counter_dict).most_common(1))
    # These are the nearest neighbors
    return (Counter(new_counter_dict).most_common(1)[0])



def tfidf(tf,df,N,weighter):

    if df == 0:
         return 0

    arg = float(N) / float(df)
    if N == 0: return 0
    if weighter == 'tfidf':
        return float(tf)*math.log(arg,10)

    return math.log(arg,10)
    return tf



def train_tfidf(words,song_dict):

    list1 = [(x,0) for x in range(0,5001)]
    #print(list1)
    df_dict = dict(list1)
    word_to_docs = dict(list1)

    for word in word_to_docs:
        word_to_docs[word] = []


    for song in song_dict:
        for tup in song_dict[song]:
            #print(type(tup[0]))
            df_dict[tup[0]] +=1
            word_to_docs[tup[0]].append(song)

    song_tfidf_dict = copy.deepcopy(song_dict)

    for song in song_tfidf_dict:
        k = 0
        for tup in song_tfidf_dict[song]:
            tup[1] = tfidf(song_dict[song][k][1],df_dict[tup[0]],len(song_dict),'tfidf')
            k+=1

    #print(song_tfidf_dict)
    return df_dict, song_tfidf_dict, word_to_docs, len(song_dict)
    #print(df_dict)
    #print(word_to_docs)

def test_tfidf(words, song_dict, df_dict, N):
    song_tfidf_dict = copy.deepcopy(song_dict)

    for song in song_tfidf_dict:
        k = 0
        for tup in song_tfidf_dict[song]:
            tup[1] = tfidf(song_dict[song][k][1],df_dict[tup[0]],N,'tfidf')
            k+=1

    return song_tfidf_dict


def main(argv):
    max_train = 2000
    max_test = 500

    filename = argv[1]
    test_filename = argv[2]
    #print(filename)

    # Create year dictionary
    id_year_filename = argv[3]

    classification_method = argv[4]

    id_year = open(id_year_filename, "r")
    year_dict = {}
    for line in id_year:
        values = line.split()
        year_dict[values[0]] = values[1]


    mxm_data = open(abspath(filename)).readlines()
    test_data = open(abspath(test_filename)).readlines()

    words = set(mxm_data[0].split(','))
    test_words = set(test_data[0].split(','))

    song_dict = {}
    label_dict = {}

    # for svm
    train_label_dict = {}
    test_label_dict = {}
    j = 0

    for line in mxm_data[1:]:
        if j >= max_train:
            break
        line = line.split(',')
        line[-1] = line[-1][:-1]
        non_float_data = [line_data.split(':') for line_data in line[2:]]
        #print(non_float_data[0][1])
        data = [[int(line_data[0]),int(line_data[1])] for line_data in non_float_data]
        # for line_data in non_float_data:
        #   print(line_data)
        #   (float(line_data[0]),float(line_data[1]))
        if line[1] in year_dict:
            label_dict[line[1]] = year_dict[line[1]][0:-1]
            train_label_dict[line[1]] = year_dict[line[1]][0:-1]

            song_dict[line[1]] = data
            j += 1

    test_label_dict = {}
    test_song_dict = {}
    j = 0

    for line in test_data[1:]:
        if j >= max_test:
            break
        line = line.split(',')
        line[-1] = line[-1][:-1]
        non_float_data = [line_data.split(':') for line_data in line[2:]]
        data = [[int(line_data[0]),int(line_data[1])] for line_data in non_float_data]
        if line[1] in year_dict:
            label_dict[line[1]] = year_dict[line[1]][0:-1]
            test_label_dict[line[1]] = year_dict[line[1]][0:-1]
            test_song_dict[line[1]] = data
            j += 1

    df_dict, song_tfidf_dict, word_to_docs, N = train_tfidf(words, song_dict)
    test_dict = test_tfidf(words, test_song_dict, df_dict, len(song_dict))

    if(classification_method == 'rocchio'):
        cosineDiff(test_dict, year_dict, song_tfidf_dict, df_dict, N)

    if(classification_method == 'svm'):
        kernel = argv[5]
        #svm.svm_main(words,test_words,song_tfidf_dict,test_dict,train_label_dict, test_label_dict)
        #svm.svm_main(song_tfidf_dict,test_dict,label_dict,test_label_dict)

    if(classification_method == 'knn'):
        k_val = int(argv[5])
        train_NN(song_tfidf_dict, test_dict, label_dict, k_val)

    if(classification_method == 'lda'):
        lda_main(words,song_dict)

    #nearestNeighbor(test_dict, df_dict, N, song_tfidf_dict, label_dict, 1)


    #for line in open(file).read():






if __name__ == "__main__":
    main(sys.argv)
