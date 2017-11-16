import pickle
import collections
import tensorflow as tf

def load_data(filename):
    data = []
    with open(filename, "r", encoding='utf-8') as f:
        data.extend(tf.compat.as_str(f.read().lower()).split())
    return data

def load_data_by_line(filename):
    data = []
    with open(filename, "r", encoding='utf-8') as f:
        for line in f:
            data.extend(line.split())
    return data

def build_train_data(data, counter, max_vocab_num):

    count = [['unk', -1]]
    count.extend(counter.most_common(max_vocab_num-1))

    word_dict = {}
    for word, _ in count:
        word_dict[word] = len(word_dict)

    id_data = []
    for word in data:
        if word in word_dict:
            id_data.append(word_dict[word])
        else:
            id_data.append(word_dict['unk'])
    reverse_dict = dict(zip(word_dict.values(), word_dict.keys()))

    return id_data, word_dict, reverse_dict