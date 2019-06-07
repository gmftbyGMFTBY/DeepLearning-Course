# coding:utf-8

import argparse
import collections
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.legacy_seq2seq as seq2seq

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BEGIN_CHAR = '['
END_CHAR = ']'
UNKNOWN_CHAR = '*'
MAX_LENGTH = 100
MIN_LENGTH = 10
max_words = 3000
epochs = 1
poetry_file = './data/poetry.txt'
save_dir = 'log'

"""
唐诗数据封装
"""


class Data:
    def __init__(self):
        self.batch_size = 64
        self.poetry_file = poetry_file
        self.load()
        self.create_batches()

    """
    读入唐诗文本数据集，生成词典
    """

    def load(self):
        def handle(line):
            if len(line) > MAX_LENGTH:
                index_end = line.rfind('。', 0, MAX_LENGTH)
                index_end = index_end if index_end > 0 else MAX_LENGTH
                line = line[:index_end + 1]
            return BEGIN_CHAR + line + END_CHAR

        self.poetrys = []
        lines = open(self.poetry_file, encoding='utf-8')
        self.poetrys = [line.strip().replace(' ', '').split(':')[1] for line in lines]
        self.poetrys = [handle(line) for line in self.poetrys if len(line) > MIN_LENGTH]
        # 所有字
        words = []
        for poetry in self.poetrys:
            words += [word for word in poetry]
        counter = collections.Counter(words)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        words, _ = zip(*count_pairs)

        # 取出现频率最高的词的数量组成字典，不在字典中的字用'*'代替
        words_size = min(max_words, len(words))
        self.words = words[:words_size] + (UNKNOWN_CHAR,)
        self.words_size = len(self.words)

        # 字映射成id
        self.char2id_dict = {w: i for i, w in enumerate(self.words)}
        self.id2char_dict = {i: w for i, w in enumerate(self.words)}
        self.unknow_char = self.char2id_dict.get(UNKNOWN_CHAR)
        self.char2id = lambda char: self.char2id_dict.get(char, self.unknow_char)
        self.id2char = lambda num: self.id2char_dict.get(num)
        self.poetrys = sorted(self.poetrys, key=lambda line: len(line))
        self.poetrys_vector = [list(map(self.char2id, poetry)) for poetry in self.poetrys]

    """
    生成批次数据
    """

    def create_batches(self):
        self.n_size = len(self.poetrys_vector) // self.batch_size
        self.poetrys_vector = self.poetrys_vector[:self.n_size * self.batch_size]
        self.x_batches = []
        self.y_batches = []
        for i in range(self.n_size):
            batches = self.poetrys_vector[i * self.batch_size: (i + 1) * self.batch_size]
            length = max(map(len, batches))
            # padding *
            for row in range(self.batch_size):
                if len(batches[row]) < length:
                    r = length - len(batches[row])
                    batches[row][len(batches[row]): length] = [self.unknow_char] * r
            xdata = np.array(batches)
            ydata = np.copy(xdata)
            ydata[:, :-1] = xdata[:, 1:]
            self.x_batches.append(xdata)
            self.y_batches.append(ydata)


"""
唐诗生成字符级RNN网络搭建
"""

class Model:
    def __init__(self, data, model='lstm', infer=False):
        self.rnn_size = 128
        self.n_layers = 2

        if infer:
            self.batch_size = 1
        else:
            self.batch_size = data.batch_size

        if model == 'rnn':
            cell_rnn = tf.nn.rnn_cell.BasicRNNCell
        elif model == 'gru':
            cell_rnn = tf.nn.rnn_cell.GRUCell
        elif model == 'lstm':
            cell_rnn = tf.nn.rnn_cell.LSTMCell
        cell = cell_rnn(self.rnn_size, name='basic_lstm_cell', )
        self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.n_layers)

        self.x_tf = tf.placeholder(tf.int32, [self.batch_size, None])
        self.y_tf = tf.placeholder(tf.int32, [self.batch_size, None])

        self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [self.rnn_size, data.words_size])
            softmax_b = tf.get_variable("softmax_b", [data.words_size])
            # with tf.device("/gpu:1"):
            embedding = tf.get_variable(
                "embedding", [data.words_size, self.rnn_size])
            inputs = tf.nn.embedding_lookup(embedding, self.x_tf)
        # tf.nn.dynamic_rnn可以运行输入的shape不同，
        # 而tf.nn.rnn必须要求输入的shape必须一致。
        outputs, final_state = tf.nn.dynamic_rnn(
            self.cell, inputs, initial_state=self.initial_state, scope='rnnlm')
        self.output = tf.reshape(outputs, [-1, self.rnn_size])
        self.logits = tf.matmul(self.output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        self.final_state = final_state
        pred = tf.reshape(self.y_tf, [-1])

        # seq2seq 损失
        loss = seq2seq.sequence_loss_by_example([self.logits],
                                                [pred],
                                                [tf.ones_like(pred, dtype=tf.float32)])

        self.cost = tf.reduce_mean(loss)
        self.learning_rate = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()

        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), clip_norm=5)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))


"""
训练函数入口
"""


def train(data, model):
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        elif len(os.listdir(save_dir)) > 0:
            model_file = tf.train.latest_checkpoint(save_dir)
            saver.restore(sess, model_file)
        n = 0
        for epoch in range(epochs):
            # 逐级降低学习率
            sess.run(tf.assign(model.learning_rate, 0.002 * (0.97 ** epoch)))
            pointer = 0
            for batch in range(data.n_size):
                n += 1
                feed_dict = {model.x_tf: data.x_batches[pointer], model.y_tf: data.y_batches[pointer]}
                pointer += 1
                train_loss, _, _ = sess.run(fetches = [model.cost, model.final_state, model.train_op], feed_dict=feed_dict)
                info = "{}/{} (epoch {}) | train_loss {:.3f}" \
                    .format(epoch * data.n_size + batch,
                            epochs * data.n_size, epoch, train_loss)
                print(info)

                # 定期保存checkpoint, 每1000步保存一次，如果是最后一步，也会保存
                if (epoch * data.n_size + batch) % 1000 == 0 \
                        or (epoch == epochs - 1 and batch == data.n_size - 1):
                    checkpoint_path = os.path.join(save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=n)
                    print("model saved to {}".format(checkpoint_path))
            print('\n')


"""
测试函数入口：默认随机生成，可设置head字符串，生成藏头诗
"""


def test(data, model, head=u''):
    def to_word(weights):
        t = np.cumsum(weights) # 前缀和
        s = np.sum(weights)    # 经过softmax，概率和为1
        sample = int(np.searchsorted(t, np.random.rand(1) * s))
        return data.id2char(sample)

    for word in head:
        if word not in data.words:
            return u'{} 不在字典中'.format(word)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())
        model_file = tf.train.latest_checkpoint(save_dir)
        # print(model_file)
        saver.restore(sess, model_file)

        if head:
            print('生成藏头诗 ---> ', head)
            poem = BEGIN_CHAR
            for head_word in head:
                poem += head_word
                x = np.array([list(map(data.char2id, poem))])
                state = sess.run(model.cell.zero_state(1, tf.float32))
                feed_dict = {model.x_tf: x, model.initial_state: state}
                [probs, state] = sess.run([model.probs, model.final_state], feed_dict)
                word = to_word(probs[-1])
                while word != u'，' and word != u'。':
                    poem += word
                    x = np.zeros((1, 1))
                    x[0, 0] = data.char2id(word)
                    [probs, state] = sess.run([model.probs, model.final_state],
                                              {model.x_tf: x, model.initial_state: state})
                    word = to_word(probs[-1])
                poem += word
            return poem[1:]
        else:
            poem = ''
            head = BEGIN_CHAR
            x = np.array([list(map(data.char2id, head))])
            state = sess.run(model.cell.zero_state(1, tf.float32))
            feed_dict = {model.x_tf: x, model.initial_state: state}
            [probs, state] = sess.run([model.probs, model.final_state], feed_dict)
            word = to_word(probs[-1])
            while word != END_CHAR:
                poem += word
                x = np.zeros((1, 1))
                x[0, 0] = data.char2id(word)
                feed_dict = {model.x_tf: x, model.initial_state: state}
                [probs, state] = sess.run([model.probs, model.final_state], feed_dict)
                word = to_word(probs[-1])
            return poem


"""
主函数：从命令行接受参数，根据参数选择模式（训练或者测试）,并执行
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test',
                        help=u'usage: train or test, test is default')
    parser.add_argument('--head', type=str, default='', help='生成藏头诗')

    args = parser.parse_args()

    if args.mode == 'train':
        data = Data()
        model = Model(data=data, infer=False)
        train(data, model)

    elif args.mode == 'test':
        data = Data()
        model = Model(data=data, infer=True)
        poem = test(data, model, head=args.head)
        print(poem)

    else:
        msg = """
                   Usage:
                   Training: 
                       python poetry_gen.py --mode train
                   Sampling:
                       python poetry_gen.py --mode test --head 明月别枝惊鹊
              """
        print(msg)

if __name__ == '__main__':
    main()
