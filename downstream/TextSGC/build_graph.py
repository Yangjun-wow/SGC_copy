import argparse
import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from tqdm import tqdm
from collections import Counter
import remove_words
import itertools
from train import args
from nltk.corpus import stopwords

parser = argparse.ArgumentParser(description='Build Document Graph')
parser.add_argument('--dataset', type=str, default='20ng',
                    choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'yelp', 'ag_news'],
                    help='dataset name')
parser.add_argument('--embedding_dim', type=int, default=300,
                    help='word and document embedding size.')
args = parser.parse_args()

# build corpus
dataset = args.dataset

# build corpus

word_embeddings_dim = args.embedding_dim
word_vector_map = {} # TODO: modify this to use embedding

# 训练集数据的id索引
train_ids = []
# 测试集是数据的id索引
test_ids = []
# 标签set集合
label_names = set()
# 训练集数据的标签索引
train_labels = []
# 测试集数据的标签索引
test_labels = []


#训练集，测试集的数据
lines_train = []
lines_test = []

with open('../../data/' + args.data_train_path,encoding='utf-8', mode='r') as f1,open('../../data/' + args.data_test_path, encoding='utf-8',mode='r') as f2:
    lines_train = f1.readlines()
    train_length = len(lines_train)
    lines_test = f2.readlines()

    label_names = ['content','generate']
    label_names_to_index = {'content':0,'generate':1}

    # 为标签创建索引,用索引表示数据标签
    for id, line in enumerate(lines_train):
        data_label_name, data_content = line.strip().split("\t")
        train_ids.append(id)
        train_labels.append(label_names_to_index[data_label_name])
    for id, line in enumerate(lines_test):
        data_label_name, data_content = line.strip().split("\t")
        test_ids.append(id + train_length)
        test_labels.append(label_names_to_index[data_label_name])



print("Loaded labels and indices")
doc_content_list = lines_train + lines_test
if args.language == 'en':
    stop_words = set(stopwords.words('english'))
    doc_content_list = remove_words.get_clean_words_en(doc_content_list,stop_words)
else:
    stop_words = remove_words.get_stop_words()
    doc_content_list = remove_words.get_clean_words_cn(doc_content_list,stop_words)

print("Loaded document content")
# 创建词汇表，显示进度-Build vocab
word_freq = Counter()
progress_bar = tqdm(doc_content_list)
progress_bar.set_postfix_str("building vocabulary")
for doc_words in progress_bar:
    words = doc_words.split()
    word_freq.update(words)

vocab, _ = zip(*word_freq.most_common())

# 创建字典，将vocab中词汇表中的单词映射到唯一的整数标识符，标识符根据单词顺序也就是词频来标
word_id_map = dict(zip(vocab, np.array(range(len(vocab)))+len(train_ids+test_ids)))
vocab_size = len(vocab)

# split training and validation
idx = list(range(len(train_labels)))
random.shuffle(idx)
train_ids = [train_ids[i] for i in idx]
train_labels = [train_labels[i] for i in idx]

idx = list(range(len(test_labels)))
random.shuffle(idx)
test_ids = [test_ids[i] for i in idx]
test_labels = [test_labels[i] for i in idx]

# Construct feature vectors
# def average_word_vec(doc_id, doc_content_list, word_to_vector):
#     doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
#     doc_words = doc_content_list[doc_id]
#     words = doc_words.split()
#     for word in words:
#         if word in word_vector_map:
#             word_vector = word_vector_map[word]
#             doc_vec = doc_vec + np.array(word_vector)
#     doc_vec /= len(words)
#     return doc_vec

# def construct_feature_label_matrix(doc_ids, doc_content_list, word_vector_map):
#     row_x = []
#     col_x = []
#     data_x = []
#     for i, doc_id in enumerate(doc_ids):
#         doc_vec = average_word_vec(doc_id, doc_content_list, word_vector_map)
#         for j in range(word_embeddings_dim):
#             row_x.append(i)
#             col_x.append(j)
#             data_x.append(doc_vec[j])
#     x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
#         real_train_size, word_embeddings_dim))
#
#     y = []
#     for label in train_labels:
#         one_hot = [0 for l in range(len(label_list))] #？？？？
#         one_hot[label] = 1
#         y.append(one_hot)
#     y = np.array(y)
#     return x, y

# not used
# train_x, train_y = construct_feature_label_matrix(train_ids, doc_content_list, word_vector_map)
# val_x, val_y = construct_feature_label_matrix(val_ids, doc_content_list, word_vector_map)
# test_x, test_y = construct_feature_label_matrix(test_ids, doc_content_list, word_vector_map)

# print("Finish building feature vectors")

# Creating word and word edges
def create_window(seq, n=2):
    """Returns a sliding window (of width n) over data from the iterable,
    code taken from https://docs.python.org/release/2.3.5/lib/itertools-example.html"""
    it = iter(seq)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

# word co-occurence with context windows
def construct_context_windows(ids, doc_words_list, window_size=20):
    windows = []
    for id in ids:
        doc_words = doc_content_list[id]
        words = doc_words.split()
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            windows += list(create_window(words, window_size))
    return windows

def count_word_window_freq(windows):
    word_window_freq = Counter()
    progress_bar = tqdm(windows)
    progress_bar.set_postfix_str("constructing context window")
    for window in progress_bar:
        word_window_freq.update(set(window))
    return word_window_freq

def count_word_pair_count(windows):
    word_pair_count = Counter()
    progress_bar = tqdm(windows)
    progress_bar.set_postfix_str("counting word pair frequency")
    for window in progress_bar:
        word_pairs = list(itertools.permutations(window, 2))
        word_pair_count.update(word_pairs)
    return word_pair_count
  
def build_word_word_graph(num_window, word_id_map, word_window_freq, word_pair_count):
    row = []
    col = []
    weight = []
    # pmi as weights
    for pair, count in word_pair_count.items():
        i, j = pair
        word_freq_i = word_window_freq[i]
        word_freq_j = word_window_freq[j]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(word_id_map[i])
        col.append(word_id_map[j])
        weight.append(pmi)
    return row, col, weight

def calc_word_doc_freq(ids, doc_content_list):
    # Count number of documents that contain a word
    word_doc_list = {} # mapping from word to document id
    word_doc_freq = Counter()
    for doc_id in ids:
        doc_words = doc_content_list[doc_id]
        words = set(doc_words.split())
        word_doc_freq.update(words)
    return word_doc_freq

def calc_doc_word_freq(ids, doc_content_list):
    doc_word_freq = Counter()
    for doc_id in ids:
        doc_words = doc_content_list[doc_id]
        words = doc_words.split()
        word_ids = [word_id_map[word] for word in words]
        doc_word_pairs = zip([doc_id for _ in word_ids], word_ids)
        doc_word_freq.update(doc_word_pairs)
    return doc_word_freq

def build_doc_word_graph(ids, doc_words_list, doc_word_freq, word_doc_freq, phase='B'):
    row = []
    col = []
    weight = []
    for i, doc_id in enumerate(ids):
        doc_words = doc_words_list[doc_id]
        words = set(doc_words.split())
        doc_word_set = set()
        for word in words:
            word_id = word_id_map[word]
            key = (doc_id, word_id)
            freq = doc_word_freq[key]
            idf = log(1.0 * len(ids) /
                      word_doc_freq[word])
            w = freq*idf
            if phase == "B":
                row.append(doc_id)
                col.append(word_id)
                weight.append(w)
            elif phase == "C":
                row.append(word_id)
                col.append(doc_id)
                weight.append(w)
            else: raise ValueError("wrong phase")
    return row, col, weight

def concat_graph(*args):
    rows, cols, weights = zip(*args)
    row = list(itertools.chain(*rows))
    col = list(itertools.chain(*cols))
    weight = list(itertools.chain(*weights))
    return row, col, weight

def export_graph(graph, node_size, phase=""):
    row, col, weight = graph
    adj = sp.csr_matrix(
        (weight, (row, col)), shape=(node_size, node_size))
    if phase == "": path = "../../data/ind.{}.adj".format(dataset)
    else: path = "../../data/ind.{}.{}.adj".format(dataset, phase)
    with open(path, 'wb') as f:
        pkl.dump(adj, f)

ids = train_ids+test_ids
windows = construct_context_windows(ids, doc_content_list)
word_window_freq = count_word_window_freq(windows)
word_pair_count = count_word_pair_count(windows)
D = build_word_word_graph(len(windows), word_id_map, word_window_freq, word_pair_count)

doc_word_freq = calc_doc_word_freq(ids, doc_content_list)
word_doc_freq = calc_word_doc_freq(ids, doc_content_list)
B = build_doc_word_graph(ids, doc_content_list, doc_word_freq, word_doc_freq, phase="B")
C = build_doc_word_graph(ids, doc_content_list, doc_word_freq, word_doc_freq, phase="C")

node_size = len(vocab)+len(train_ids)+len(test_ids)
export_graph(concat_graph(B, C, D), node_size, phase="BCD")
export_graph(concat_graph(B, C), node_size, phase="BC")
export_graph(concat_graph(B, D), node_size, phase="BD")
export_graph(B, node_size, phase="B")

# dump objects
f = open("../../data/ind.{}.{}.x".format(dataset, "train"), 'wb')
pkl.dump(train_ids, f)
f.close()

f = open("../../data/ind.{}.{}.y".format(dataset, "train"), 'wb')
pkl.dump(train_labels, f)
f.close()

f = open("../../data/ind.{}.{}.x".format(dataset, "test"), 'wb')
pkl.dump(test_ids, f)
f.close()

f = open("../../data/ind.{}.{}.y".format(dataset, "test"), 'wb')
pkl.dump(test_labels, f)
f.close()
