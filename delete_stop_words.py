# coding utf-8

import os

label = ['财经', '房产', '健康', '教育', '军事', '科技', '体育', '娱乐', '证券']


# stop_words为停用词表
stop_words = []
with open("停用词.txt", 'r', encoding='utf-8') as sf:
    strings = sf.readlines()
    for i in range(len(strings)):
        strings[i] = strings[i][:-1]
stop_words = strings

name = "数据/"
# 处理数据集, 将‘数据’各个txt文档中出现的停用词删去
for l in label:
    filen = name + l
    filen = filen + '/'
    files = os.listdir(filen)
    for f in files:
        tmpef = filen + f
        # tmps是txt文件读到内存的名称
        tmps = []
        try:
            tmpf = open(tmpef, 'r', encoding='utf-8')
            strings1 = tmpf.readlines()
            tmps = strings1[0]
            tmpf.close()
            tmps = tmps.split()
            i = 0
            while i < len(tmps):
                che = False
                word = tmps[i]
                for j in stop_words:
                    if j == word:
                        che = True
                        break
                if word.isdigit():
                    che = True
                if che:
                    tmps.pop(i)
                    i -= 1
                i += 1
        except:
            tmps = ['我', '谈谈', '发生', '身上', '囧事', '捡', '皮球', '时', '闪了', '腰']
        with open(tmpef, 'w', encoding='utf-8') as tmpf:
            for i in tmps:
                tmpf.write(i)
                tmpf.write(' ')
            tmpf.write('\n')

