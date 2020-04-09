#encoding:utf-8


class pm(object):
    # 参数
    embedding_dim = 100      # 词向量维度
    num_classes = 9        # 类别数
    vocab_size = 9561       # 词汇表达小
    pre_trianing = None      # use word_vector.txt to initiate
    max_pad = 52            # 最长的句子真实长度并作为rnn循环的长度，由于采用动态rnn，
                            # 真实运行时被每个batch的最长句子长度代替，这里仅作为初始化
    num_layers = [60, 60]          # lstm层数与hidden维度
    hidden = 60       # 隐藏层维度

    keep_prob = 0.5        # 保留比例
    learning_rate = 0.01    # 学习率
    clip = 5.0
    lr_decay = 0.9           # learning rate decay
    batch_size = 64          # 每批训练大小
    num_epochs = 3           # 总迭代轮次





