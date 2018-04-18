import sys, math, heapq

LABEL_FILE_NAME = 'MSD_track_id_and_year.txt'

# HELPERS FOR print_stats
def handle_mxm_line_gen():
    id_map = load_years()
    def handle_mxm_line(line):
        fields = line.split(',')
        mxm_id = fields[1]
        return decade_from_id_map(mxm_id, id_map)
    return handle_mxm_line

def handle_label_line(line):
    _, year = line.split('\t')
    decade = year[:3]
    return decade

# STATS
def print_stats(file_name, max_songs):
    decades = dict()
    total = 0
    handle_line = handle_label_line
    if file_name != LABEL_FILE_NAME:
        handle_line = handle_mxm_line_gen()
        file = open(file_name)
        # don't need the top 5000 words
        file.readline()
    else:
        file = open(file_name)
    lines = file.readlines()
    file.close()
    max_songs = min(len(lines), max_songs)
    i = 0
    while total < max_songs and i < len(lines):
        line = lines[i]
        i += 1
        decade = handle_line(line)
        if decade == '000':
            continue
        elif decade not in decades:
            decades[decade] = 0
        decades[decade] += 1
        total += 1

    print('\ntotal # of songs: {}\n'.format(total))
    for decade, count in sorted(decades.items(), key=lambda x:x[0], reverse=True):
        percent = '{:7.3f}%'.format((count / total) * 100)
        count_str = '{:7d}\t'.format(count)
        decade_str = '{:}0s: '.format(decade)
        print(decade_str, count_str, percent)

# WORDS
def get_top_words(file_name, num_words):
    if file_name == LABEL_FILE_NAME:
        sys.exit('This file is not compatible with the topics feature.')
    id_map = load_years()
    decade_map = dict() # map from decade to dict(top_words)
    # for max_songs in decade, count instances of top words in dict

    # get 5000 words and all song lines
    file = open(file_name)
    word_array = file.readline().split(',')
    # iterate through all songs, stop at max_songs
    lines = file.readlines()
    file.close()

    # get stopwords
    stopword_file = open('stopwords')
    stopwords = set([word.strip() for word in stopword_file.readlines()])
    stopword_file.close()

    # get stem_map
    stem_map_file = open('mxm_reverse_mapping.txt')
    stem_map = dict()
    for line in stem_map_file.readlines():
        word_list = line.split('<SEP>')
        # stemmed word => unstemmed word
        stemmed = word_list[0].strip()
        unstemmed = word_list[1].strip()
        stem_map[stemmed] = unstemmed

    for line in lines:
        line = line.strip().split(',')
        mxm_id = line[1]
        decade = decade_from_id_map(mxm_id, id_map)
        if decade == '000':
            continue
        song_words = [tuple(t.split(':')) for t in line[2:]]
        top_words = []
        # using heap, find top num_words
        for word, freq_str in song_words:
            word_index = int(word) - 1
            freq = int(freq_str)
            if len(top_words) < num_words:
                heapq.heappush(top_words, (freq, word_index))
            elif top_words[0][0] < freq:
                heapq.heapreplace(top_words, (freq, word_index))

        # add the top words to the respective decade
        if decade not in decade_map:
            decade_map[decade] = dict()
        for _, word_index in top_words:
            curr_word_map = decade_map[decade]
            if word_index not in curr_word_map:
                curr_word_map[word_index] = 0
            curr_word_map[word_index] += 1

    # use tfidf instead of raw tf
    df_map = dict()
    for decade, curr_word_map in decade_map.items():
        for index, freq in curr_word_map.items():
            if index not in df_map:
                df_map[index] = 0
            df_map[index] += 1
    N = len(decade_map)
    for decade, curr_word_map in decade_map.items():
        for index, freq in curr_word_map.items():
            tfidf = freq * math.log(N / (df_map[index] + 1))
            curr_word_map[index] = tfidf

    # filter stopwords, sort, and print results
    for decade, curr_word_map in sorted(decade_map.items(), key=lambda x:int(x[0])):
        top_words = sorted(curr_word_map.items(), key=lambda x:x[1], reverse=True)
        print(decade)
        count = 0
        for index, freq in top_words:
            word = word_array[index]
            if word in stopwords:
                continue
            print('\t{}'.format(stem_map[word]))
            count += 1
            if count > num_words:
                break

def load_years():
    id_map = dict()
    label_file = open(LABEL_FILE_NAME)
    lines = label_file.readlines()
    label_file.close()
    for line in lines:
        mxm_id, year = line.split('\t')
        id_map[mxm_id] = year
    return id_map

def decade_from_id_map(mxm_id, id_map):
    year = '0000' if mxm_id not in id_map else id_map[mxm_id]
    decade = year[:3]
    return decade

def main(argv):
    # max_train = 2000
    # max_test = 500
    max_songs = 175234
    max_words = 20
    # python3 song_insight.py <filename> <'stats' | 'words'> <max_songs | max_words>
    if len(argv) < 2:
        sys.exit("Please specify input file name")
    file_name = argv[1]
    if len(argv) < 3:
        sys.exit("Please specify program option: 'stats' or 'words'")
    # specify stats or words
    option = argv[2]
    if option == 'stats':
        function = print_stats
        max_val = max_songs
    elif option == 'words':
        function = get_top_words
        max_val = max_words
    else:
        sys.exit("Program options are: 'stats' or 'words'")

    # specify maximum number of songs to analyze
    # for training subset:  2000
    # for testing subset:   500
    if len(argv) >= 4:
        max_val = int(argv[3])

    function(file_name, max_val)

if __name__ == "__main__":
    main(sys.argv)
