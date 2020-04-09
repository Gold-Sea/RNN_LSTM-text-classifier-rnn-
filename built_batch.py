# coding utf-8
import os
import numpy as np
import tensorflow as tf
import data_handle as gr
import tensorflow.contrib.keras as kr


# 提取高频出现的前10000个词，并从词向量库中提取词向量的值并保存到word_vector.txt文件中，建立的词表大小为9561
# 该函数已经预处理好了不用再次运行
def built_vocab_vector():
    cur = 0
    word_list = []
    num_classes = 9
    for i in gr.all_sec:
        for j in i:
            word_list.append(j)
    word_sum = {}
    for i in word_list:
        if i in word_sum.keys():
            word_sum[i] += 1
        else:
            word_sum[i] = 1
    word_sum = sorted(word_sum.items(), key=lambda d: d[1], reverse=True)
    words = []
    # 只选取出现频率最高的10000个词
    for i in range(10000):
        words.append(word_sum[i][0])
    word_dict = {w: i for i, w in enumerate(words)}
    number_dict = {i: w for i, w in enumerate(words)}
    n_class = len(words)  # number of Vocabulary
    # 该文件路径是在网上下载的预处理的词向量
    with open('C:/Users/86183/Desktop/新建文件夹/vector_word.txt', encoding='utf-8') as file:
        total = file.readlines()
        total_word = []
        total_vec = np.zeros(shape=[370695, 100], dtype=np.float)
        for i in range(370695):
            sec = total[i].split()
            total_word.append(sec[0])
            for k in range(100):
                total_vec[i][k] = float(sec[k + 1])
        total_dict = {w: i for i, w in enumerate(total_word)}
    # 新的词表
    word_list = []
    word_index = []
    sum = 0
    # 将处理好的词向量写入"word_vector.txt"
    for i in words:
        if i in total_word:
            word_list.append(i)
            word_index.append(total_dict[i])
            sum += 1
    embedding = np.zeros(shape=[sum, 100], dtype=np.float)
    for i in range(sum):
        embedding[i] = total_vec[word_index[i]]
    with open('word_vector.txt', 'w', encoding='utf-8') as file:
        for i in range(sum):
            file.write(str(word_list[i]))
            file.write('\n')
            for j in range(100):
                file.write(str(embedding[i][j]))
                file.write('\n')


# 建立四个数据结构，分别为这9561个单词的总列表，词向量，词-number的字典，number-词的字典
def built_word_dict():
    word_list = []
    word_vec = np.zeros(shape=[9561, 100], dtype=np.float)
    with open('word_vector.txt', 'r', encoding='utf-8') as file:
        total = file.readlines()
        for i in range(9561):
            word_list.append(total[i * 101][:-1])
            for j in range(100):
                word_vec[i][j] = float(total[i * 101 + j + 1][:-1])
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    return word_list, word_vec, word_dict, number_dict


word_list, word_vec, word_dict, number_dict = built_word_dict()


# 返回5个数据结构分别是训练集的句向量和标签，以及测试集的句向量和标签,句子向量要pad的长度
def built_minst():
    max_padding = 0
    # 删除十个训练集宇测试集中不在word_list出现的单词，并新建这些数据集
    all_sec = []
    for j in range(gr.all_total):
        tmpe_str = []
        for string in gr.all_sec[j]:
            if string in word_list:
                tmpe_str.append(string)
        max_padding = max(max_padding, len(tmpe_str))
        all_sec.append(tmpe_str)
    all_vec = np.zeros(shape=[gr.all_total, max_padding], dtype=np.int)
    for i in range(gr.all_total):
        index = 0
        for j in all_sec[i]:
            all_vec[i][index] = word_dict[j]
            index += 1
    # 十折交叉下的训练集与测试集及其标签, 10个训练集、测试集的label是一样的
    test_vec = np.zeros(shape=[10, gr.test_total, max_padding], dtype=np.int)
    test_label = np.zeros(shape=[gr.test_total, 9], dtype=np.int)
    train_vec = np.zeros(shape=[10, gr.total, max_padding], dtype=np.int)
    train_label = np.zeros(shape=[gr.total, 9], dtype=np.int)
    for i in range(10):
        tmp1 = []
        tmp2 = []
        for j in range(9):
            num = gr.All_SIZE[j] // 10
            if i == 0:
                gr.SIZE[j] = gr.sum_SIZE[j] - gr.sum_SIZE[j - 1] - num
                gr.test_SIZE[j] = num
            tmp1.extend(all_vec[gr.sum_SIZE[j - 1] + i * num:gr.sum_SIZE[j - 1] + i * num + num])
            tmp2.extend(all_vec[gr.sum_SIZE[j - 1]: gr.sum_SIZE[j - 1] + i * num])
            tmp2.extend(all_vec[gr.sum_SIZE[j - 1] + i * num + num:gr.sum_SIZE[j]])
        test_vec[i] = tmp1
        train_vec[i] = tmp2
    for i in range(gr.total):
        train_label[i][gr.getindex(i, gr.SIZE)] = 1
    for i in range(gr.test_total):
        test_label[i][gr.getindex(i, gr.test_SIZE)] = 1
    return train_vec, train_label, test_vec, test_label, max_padding


train_vec, train_label, test_vec, test_label, max_pad = built_minst()


# 返回十折交叉中第十轮的训练集和标签
def process(cur):
    x_pad = train_vec[cur]
    y_pad = train_label
    return x_pad, y_pad


# 返回十折交叉中第十轮的测试集和标签
def process_test(cur):
    x_pad = test_vec[cur]
    y_pad = test_label
    return x_pad, y_pad


# 创建batch
def batch_iter(x, y,  batch_size=64):
    data_len = len(x)
    num_batch = int((data_len - 1)/batch_size)
    ran = np.random.permutation(np.arange(data_len))
    x_shuff = x[ran]
    y_shuff = y[ran]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i+1) * batch_size, data_len)
        yield x_shuff[start_id:end_id], y_shuff[start_id:end_id]


# 求解每一批batch最大的真实长度
def sequence(x_batch):
    seq_len = []
    for line in x_batch:
        length = np.sum(np.sign(line))
        seq_len.append(length)
    return seq_len


