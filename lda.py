
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np




def gen_feature_matrix(song_score_dict,label_dict):

    feature_matrix = np.zeros((len(song_score_dict),5001))
    song_names = [name for name in song_score_dict]
    labels = [None]*len(song_names)
    
    i = 0
    for song in song_names:
        labels[i] = label_dict[song]
        i +=1

    for i in range(len(song_names)):

        # if a word isn't in a song, the score will stay zero
        for tup in song_score_dict[song_names[i]]:
            #feature_matrix[i][tup[0]] = song_score_dict[song_names[i]]
            #print(tup[0])
            feature_matrix[i][tup[0]] = tup[1]

    return feature_matrix, labels

#song_dict is a tf_dict
def lda_main(words,song_dict,label_dict):
	X,labels = gen_feature_matrix(song_dict,label_dict)
	lda(X,labels,list(words))


def lda(X,labels,words):
    lda = LatentDirichletAllocation()
    lda.fit(X)
    doc_topic_matrix = lda.transform(X)
    decade_list = ['192','193','194','195','196','197','198','199','200']
    topic_probs_per_decade = np.zeros((9,10))
    
    i = 0 
    for decade in decade_list:
        j = 0
        #print('working with songs from decade ', decade, '...')
        for doc in doc_topic_matrix:
            if decade == labels[j]:
                #adding probs by decade
                topic_probs_per_decade[i] += np.array(doc)
                j+=1
        print("probs by topic for decade: ", decade)
        print(topic_probs_per_decade[i])
        i+=1

    print_top_words(lda, words, 10)


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()





























