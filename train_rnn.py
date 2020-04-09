import os
import tensorflow as tf
import numpy as np
import parameters_handle as ph
import built_batch as bb
import rnn_model


acc_list = []
def train(cur):
    tf.reset_default_graph()
    model = rnn_model.RnnModel()
    print('十折交叉循环第', i + 1, '次训练：')
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    x_train, y_train = bb.process(cur)
    x_test, y_test = bb.process_test(cur)
    acc = 0
    # 对训练集进行三次迭代
    for epoch in range(pm.num_epochs):
        print('Epoch:', epoch+1)
        # 构建数据集的batch
        batch_train = bb.batch_iter(x_train, y_train, batch_size=pm.batch_size)
        for x_batch, y_batch in batch_train:
            # 获取每个batch中句子的真实最大长度
            max_len = bb.sequence(x_batch)
            # 喂入数据
            feed_dict = model.feed_data(x_batch, y_batch, max_len, pm.keep_prob)
            # 进行训练并得到一些训练参数
            _, global_step, train_loss, train_accuracy = session.run([model.optimizer, model.global_step,
                                                                     model.loss, model.accuracy], feed_dict=feed_dict)
            # 达到训练次数阈值便进行一次训练集的正确率与损失函数求解
            if global_step % 50 == 0:
                test_loss, test_accuracy = model.evaluate(session, x_test, y_test)
                acc = max(acc, test_accuracy)
                print('global_step:', global_step, 'train_loss:', train_loss, 'train_accuracy:', train_accuracy,
                      'test_loss:', test_loss, 'test_accuracy:', test_accuracy)
        # 让学习率不断地衰减
        pm.learning_rate *= pm.lr_decay
    acc_list.append(acc)


pm = ph.pm
pm.pre_trianing = bb.word_vec

# 开始十折交叉循环
for i in range(10):
    train(i)
for i in range(10):
    print('十折交叉循环第', i + 1, '次正确率：', acc_list[i])
print('十折交叉循环平均正确率', np.sum(acc_list) / 10)

