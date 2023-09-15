import re
import jieba
from train import args
# nltk.download('punkt')
from collections import Counter

# 使用正则清理数据集
def clean_str_en(string):
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


#获取停止词
def get_stop_words():
    file_object = open('../../data/stopwords/cn_stopwords.txt', encoding='utf-8')
    stop_words = []
    for line in file_object.readlines():
        line = line[:-1]
        line = line.strip()
        stop_words.append(line)
    return stop_words

def get_clean_words_en(docs,stop_words):
    clean_words = []
    for doc in docs:
        if args.dataset != "mr":
            temp = clean_str_en(doc).split()
            temp = list(filter(lambda x: x not in stop_words, temp))
        else:
            temp = clean_str_en(doc).split()
        clean_words.append(temp)
    return clean_words


def get_clean_words_cn(docs,stop_words):
    clean_words = []
    for doc in docs:
        temp = clean_str_cn(doc)
        temp = cut(temp, stop_words)
        clean_words.append(temp)
    return clean_words

# jieba分词
def cut(sentence, stopwords):
    return [token for token in jieba.lcut(sentence) if token not in stopwords]


# clean_words = []
# doc_content_list = []
# # 统计词频，建立词汇表和对应的词频
# word_freq = Counter()
# for doc_words in clean_words:
#     word_freq.update(doc_words)
#
# vocab, count = zip(*word_freq.most_common())
#
# if True:
#     cutoff = -1
# else:
#     cutoff = count.index(5)
#
# vocab = set(vocab[:cutoff])
#
# # 去除低频词后再次清洗
# clean_docs = []
# for words in clean_words:
#     closed_words = [w for w in words if w in vocab]
#     doc_str = ' '.join(closed_words)
#     clean_docs.append(doc_str)


# 统计清洗好的文本的长度
# min_len = 10000
# aver_len = 0
# max_len = 0
#
# for line in clean_docs:
#     line = line.strip()
#     temp = line.split()
#     aver_len = aver_len + len(temp)
#     if len(temp) < min_len:
#         min_len = len(temp)
#     if len(temp) > max_len:
#         max_len = len(temp)
# f.close()
# aver_len = 1.0 * aver_len / len(lines)
# print('min_len : ' + str(min_len))
# print('max_len : ' + str(max_len))
# print('average_len : ' + str(aver_len))
