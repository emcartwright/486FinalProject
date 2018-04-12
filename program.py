import os,sys
import numpy as np
import os
from os.path import abspath
# from tokenizer import *
# import Stemmer
import time
import copy
import math
from collections import Counter

def nearestNeighbor(query_dict, df_dict, N, song_dict):
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

    print(dict(Counter(tfidf_vals).most_common(5)))
    # These are the nearest neighbors

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

    print(open(abspath(filename)).readlines()[0])

    mxm_data = open(abspath(filename)).readlines()

    words = set(mxm_data[0].split(','))

    song_dict = {}
    for line in mxm_data[1:]:
        line = line.split(',')
        line[-1] = line[-1][:-1]
        non_float_data = [line_data.split(':') for line_data in line[2:]]
        #print(non_float_data[0][1])
        data = [[int(line_data[0]),int(line_data[1])] for line_data in non_float_data]
        # for line_data in non_float_data:
        #   print(line_data)
        #   (float(line_data[0]),float(line_data[1]))
        song_dict[line[1]] = data




    test_dict = song_dict.pop('4168849')

    train_labels = range(0,len(song_dict))

    #print(test_dict)
    #print(song_dict)
    print("HI")
    print(test_dict)

    df_dict, song_tfidf_dict, word_to_docs, N = train_tfidf(words, song_dict,train_labels)

    print("YO")
    print(test_dict)

    nearestNeighbor(test_dict, df_dict, N, song_tfidf_dict)


    #for line in open(file).read():








if __name__ == "__main__":
    main(sys.argv)
