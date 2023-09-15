from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
# from utils import clean_str, loadWord2Vec
import sys
import argparse
import random
import re
import jieba
from collections import Counter
from train import args
# nltk.download('punkt')

def clean_str(string):
    string = re.sub(r'[?|$|.|!]', r'', string)
    string = re.sub(r'[^a-zA-Z0-9 ]', r'', string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def clean_str_cn(string):
    # 去除标点符号
    string = re.sub(r'[，。？！]', r'', string)
    # 去除非中文字符和数字
    string = re.sub(r'[^\u4e00-\u9fa5]', r'', string)
    # 添加空格分隔符
    # string = re.sub(r'([\u4e00-\u9fa5])', r' \1 ', string)
    # 去除多余空格
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def get_stop_words():
    file_object = open('../../data/stopwords/cn_stopwords.txt', encoding='utf-8')
    stop_words = []
    for line in file_object.readlines():
        line = line[:-1]
        line = line.strip()
        stop_words.append(line)
    return stop_words


def cut(sentence, stopwords):
    return [token for token in jieba.lcut(sentence) if token not in stopwords]


if args.language == 'en':
    stop_words = set(stopwords.words('english'))
else:
    stop_words = get_stop_words()

train_val_ids = []
test_ids = []

# 创建训练集，验证集，测试集的索引
with open('../../data/' + dataset + '.txt', 'r') as f:
    lines = f.readlines()
    for id, line in enumerate(lines):
        _, data_name, data_label = line.strip().split("\t")
        if data_name.find('test') != -1:
            test_ids.append(id)
        elif data_name.find('train') != -1:
            train_val_ids.append(id)

idx = list(range(len(train_val_ids)))
random.shuffle(idx)
train_val_ids = [train_val_ids[i] for i in idx]

idx = list(range(len(test_ids)))
random.shuffle(idx)
test_ids = [test_ids[i] for i in idx]

train_val_size = len(train_val_ids)
val_size = int(0.1 * train_val_size)
train_size = train_val_size - val_size
train_ids, val_ids = train_val_ids[:train_size], train_val_ids[train_size:]

doc_content_list = []
f = open('../../data/corpus/' + dataset + '.txt', 'rb')
for line in f.readlines():
    doc_content_list.append(line.strip().decode('utf-8'))
f.close()

# 存储索引
with open('../../data/ind.train.ids', "w") as f:
    f.write('\n'.join([str(i) for i in train_ids]))
with open('../../data/ind.val.ids', "w") as f:
    f.write('\n'.join([str(i) for i in val_ids]))
with open('../../data/ind.test.ids', "w") as f:
    f.write('\n'.join([str(i) for i in test_ids]))


# 数据清洗，停止词，正则
def get_clean_words(docs):
    clean_words = []
    for doc in docs:
        if args.dataset != "mr":
            temp = clean_str(doc).split()
            temp = list(filter(lambda x: x not in stop_words, temp))
        else:
            temp = clean_str(doc).split()
        clean_words.append(temp)
    return clean_words


def get_clean_words_cn(docs):
    clean_words = []
    for doc in docs:
        temp = clean_str_cn(doc)
        # temp = cut(temp, stop_words)
        clean_words.append(temp)
    return clean_words


clean_words = get_clean_words_cn(doc_content_list)

# 统计词频，建立词汇表和对于的词频
word_freq = Counter()
# total = 0
for i in train_ids + test_ids + val_ids:
    doc_words = clean_words[i]
    word_freq.update(doc_words)

vocab, count = zip(*word_freq.most_common())
if dataset == "mr" or dataset == "text_generated":
    cutoff = -1
else:
    cutoff = count.index(5)

vocab = set(vocab[:cutoff])

# 去除低频词后再次清洗
clean_docs = []
for words in clean_words:
    closed_words = [w for w in words if w in vocab]
    doc_str = ' '.join(closed_words)
    clean_docs.append(doc_str)

clean_corpus_str = '\n'.join(clean_docs)

# 写入清洗完成的数据
f = open('../../data/corpus/' + dataset + '.clean.txt', 'w')
f.write(clean_corpus_str)
f.close()

dataset = args.dataset
min_len = 10000
aver_len = 0
max_len = 0

# 统计清洗好的文本的长度
f = open('../../data/corpus/' + dataset + '.clean.txt', 'r')
lines = f.readlines()
for line in lines:
    line = line.strip()
    temp = line.split()
    aver_len = aver_len + len(temp)
    if len(temp) < min_len:
        min_len = len(temp)
    if len(temp) > max_len:
        max_len = len(temp)
f.close()
aver_len = 1.0 * aver_len / len(lines)
print('min_len : ' + str(min_len))
print('max_len : ' + str(max_len))
print('average_len : ' + str(aver_len))
