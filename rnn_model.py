import tensorflow as tf
import parameters_handle as ph
import built_batch as bb


class RnnModel(object):
    # 初始化一些基本的神将网络节点
    def __init__(self):
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, ph.pm.max_pad], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, ph.pm.num_classes], name='input_y')
        self.num_steps = tf.placeholder(dtype=tf.int32, shape=[None], name='num_steps')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.rnn_lstm()

    def rnn_lstm(self):
        # 将词向量嵌入句向量
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embedding = tf.get_variable('embedding', shape=[ph.pm.vocab_size, ph.pm.embedding_dim],
                                        initializer=tf.constant_initializer(ph.pm.pre_trianing))
            self.embedding_input = tf.nn.embedding_lookup(embedding, self.input_x)


        # 构建lstm节点
        with tf.name_scope('lstm_cell'):
            def create_cell(hidden):
                cell = tf.nn.rnn_cell.LSTMCell(hidden)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                return cell
            # lstm的深度有两层
            cells = [create_cell(_) for _ in ph.pm.num_layers]
            Cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)


        with tf.name_scope('rnn'):
            # 隐藏第一层 输入是[batch_size, seq_length, hidden_dim]
            # lstm中权重矩阵[batch_size, hidden_dim + embedding dim]，为了使第二层lstm矩阵matmul正确：hidden=embedding
            # 构建动态的rnn网络并得到h和output
            output, _ = tf.nn.dynamic_rnn(cell=Cell, inputs=self.embedding_input,
                                          sequence_length=self.num_steps, dtype=tf.float32)
            output = tf.reduce_sum(output, axis=1)
            # output:[batch_size, hidden_dim]


        with tf.name_scope('dropout'):
            # 初始化神经元正常工作的概率
            self.out_drop = tf.nn.dropout(output, keep_prob=self.keep_prob)


        with tf.name_scope('output'):
            w = tf.Variable(tf.truncated_normal([ph.pm.hidden, ph.pm.num_classes], stddev=0.1), name='w')
            b = tf.Variable(tf.constant(0.1, shape=[ph.pm.num_classes]), name='b')
            self.logits = tf.matmul(self.out_drop, w) + b
            # 预期的结果
            self.predict = tf.argmax(tf.nn.softmax(self.logits), 1, name='predict')


        with tf.name_scope('loss'):
            # 损失函数的定义
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)


        with tf.name_scope('optimizer'):
            # optimizer = tf.train.AdamOptimizer(ph.pm.learning_rate).minimize(self.loss)
            # 定义自己的优化梯度函数并进行global_step的更新（该参数与梯度的更新方法我参考助教上课时在seq2seq模型中的所讲）

            optimizer = tf.train.AdamOptimizer(ph.pm.learning_rate)
            # 计算变量梯度，得到梯度值,变量
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            # 进行梯度裁剪
            gradients, _ = tf.clip_by_global_norm(gradients, ph.pm.clip)
            # global_step 自动+1
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)


        with tf.name_scope('accuracy'):
            # 正向传播结果与真实结果相比较下正确率的求解
            correct_nums = tf.equal(self.predict, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_nums, tf.float32), name='accuracy')

    # 进行投喂数据的函数
    def feed_data(self, x_batch, y_batch, seq_len, keep_prob):
        feed_dict = {self.input_x: x_batch,
                     self.input_y: y_batch,
                     self.num_steps: seq_len,
                     self.keep_prob: keep_prob}
        return feed_dict

    # 评估函数， 得到损失函数值与正确率
    def evaluate(self, sess, x, y):
        batch_test = bb.batch_iter(x, y, ph.pm.batch_size)
        for x_batch, y_batch in batch_test:
            seq_len = bb.sequence(x_batch)
            feet_dict = self.feed_data(x_batch, y_batch, seq_len, 1.0)
            loss, accuracy = sess.run([self.loss, self.accuracy], feed_dict=feet_dict)
        return loss, accuracy
