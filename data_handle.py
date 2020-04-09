# coding utf-8

import os
import numpy as np
import tensorflow as tf
from string import digits


# 读取文档并split的函数
def read_weibo(filename):
    try:
        f = open(filename, 'r', encoding='utf-8')
        strings = f.readlines()
        tmps = strings[0]
        f.close()
        tmps = tmps.split()
        return tmps
    # 发现有一个文档错误编码，人为的理解文本应该如下
    except:
        return ['我', '谈谈', '发生', '身上', '囧事', '捡', '皮球', '时', '闪了', '腰']


# 每一轮的数据都是整合在一起的， 通过其下标判断他的标签， s是预处理的各个标签集合的元素个数
# 提取标签
def getindex(numb, s):
    ss = np.zeros(shape=9, dtype=np.int)
    ss[0] = s[0]
    for i in range(1, 9):
        ss[i] = s[i]
        ss[i] += ss[i - 1]
    res = 0
    for i in range(9):
        if numb < ss[i]:
            res = i
            break
    return res


label = ['财经', '房产', '健康', '教育', '军事', '科技', '体育', '娱乐', '证券']
# 分别是整体的文本样本以及十折交叉的十个循环文本容量大小(每一个循环都一样)
All_SIZE = np.zeros(shape=9, dtype=np.int)
SIZE = np.zeros(shape=9, dtype=np.int)
test_SIZE = np.zeros(shape=9, dtype=np.int)

# 分别是整体的文本样本以及十折交叉的十个循环文本(sec为训练集, test_sec为测试集)
all_sec = []
test_sec = [[], [], [], [], [], [], [], [], [], []]
sec = [[], [], [], [], [], [], [], [], [], []]
# 分别是整体的文本样本大小以及十折交叉的十个循环中训练集与测试集的大小(每一个循环都一样)
all_total = 0
total = 0
test_total = 0

# All_SIZE的前k和
sum_SIZE = np.zeros(shape=10, dtype=np.int)
count = 0


name = "数据/"
# 处理数据集
for l in label:
    filen = name + l
    filen = filen + '/'
    files = os.listdir(filen)
    for f in files:
        All_SIZE[count] += 1
        tmpef = filen + f
        all_sec.append(read_weibo(tmpef))
    count += 1

# 计算sum_SIZE
sum_SIZE[0] = All_SIZE[0]
for i in range(1, 9):
    sum_SIZE[i] = All_SIZE[i]
    sum_SIZE[i] += sum_SIZE[i - 1]
sum_SIZE[9] = 0
# 计算all_total
for i in All_SIZE:
    all_total += i
# 创建十折交叉数据集
for i in range(10):
    tmp1 = []
    tmp2 = []
    for j in range(9):
        num = All_SIZE[j] // 10
        if i == 0:
            SIZE[j] = sum_SIZE[j] - sum_SIZE[j - 1] - num
            test_SIZE[j] = num
        tmp1.append(all_sec[sum_SIZE[j - 1] + i * num:sum_SIZE[j - 1] + i * num + num])
        tmp2.append(all_sec[sum_SIZE[j - 1]: sum_SIZE[j - 1] + i * num])
        tmp2.append(all_sec[sum_SIZE[j - 1] + i * num + num:sum_SIZE[j]])
    for tmpe1 in tmp1:
        for tmpe2 in tmpe1:
            test_sec[i].append(tmpe2)
    for tmpe1 in tmp2:
        for tmpe2 in tmpe1:
            sec[i].append(tmpe2)

for j in SIZE:
    total += j
for j in test_SIZE:
    test_total += j
