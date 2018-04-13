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

from collections import Counter



def train_NN(train_queries, df_dict, N, song_dict):
    for query_dict in train_queries:
        five_nearest = nearestNeighbor(query_dict,df_dict,N, song_dict)




def nearestNeighbor(query_dict, df_dict, N, song_dict, label_dict, k):
    for tup in query_dict:
        tup[1] = tfidf(tup[1],df_dict[tup[0]],N,'tfidf')

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

    print(dict(Counter(tfidf_vals).most_common(k)))
    best_vals = Counter(tfidf_vals).most_common(k)

    new_counter_dict = {}
    curVal = 2000
    for val in best_vals:
        print(val[0])
        #exit(1)
        if(label_dict[val[0]] in new_counter_dict):
            new_counter_dict[label_dict[val[0]]] += curVal
        else:
            new_counter_dict[label_dict[val[0]]] = curVal
        curVal -=1

    print(Counter(new_counter_dict).most_common(1))
    # These are the nearest neighbors
    return Counter(new_counter_dict).most_common(1)[0]



def tfidf(tf,df,N,weighter):

    if df == 0:
        return 0

    arg = float(N) / float(df)
    if N == 0: return 0
    if weighter == 'tfidf':
        return float(tf)*math.log(arg,10)

    return math.log(arg,10)



def train_tfidf(words,song_dict, train_labels):

    list1 = [(x,0) for x in range(0,5000)]
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




def main(argv):

    filename = argv[1]
    #print(filename)

    # Create year dictionary
    id_year_filename = argv[2]
    print(id_year_filename)
    id_year = open(id_year_filename, "r")
    year_dict = {}
    for line in id_year:
        values = line.split()
        year_dict[values[0]] = values[1]


    print(open(abspath(filename)).readlines()[0])

    mxm_data = open(abspath(filename)).readlines()

    words = set(mxm_data[0].split(','))

    song_dict = {}
    label_dict = {}
    for line in mxm_data[1:]:
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
            song_dict[line[1]] = data
        #label_dict[line[1]] = year_dict[line[0]]

    print(label_dict)

    test_dict = song_dict.pop('851082')

    #delete me
    labels = np.random.randint(1,4,len(song_dict))

    df_dict, song_tfidf_dict, word_to_docs, N = train_tfidf(words, song_dict,labels)


    nearestNeighbor(test_dict, df_dict, N, song_tfidf_dict, label_dict, 1)

    svm.svm_main(song_tfidf_dict,label_dict)



    #for line in open(file).read():






if __name__ == "__main__":
    main(sys.argv)
