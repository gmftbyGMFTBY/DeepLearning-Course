#!/usr/bin/python
import tensorflow as tf
import os
from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string('mode', 'train',
                       'The mode can be train, eval or test')

tf.flags.DEFINE_boolean('load', False,
                        'Turn on to load a pretrained model from either \
                        the latest checkpoint or a specified file')

tf.flags.DEFINE_string('model_file', None,
                       'If sepcified, load a pretrained model from this file')

tf.flags.DEFINE_boolean('load_cnn', False,
                        'Turn on to load a pretrained CNN model')

tf.flags.DEFINE_string('cnn_model_file', './vgg16_no_fc.npy',
                       'The file containing a pretrained CNN model')

tf.flags.DEFINE_boolean('train_cnn', True,
                        'Turn on to train both CNN and RNN. \
                         Otherwise, only RNN is trained')

tf.flags.DEFINE_integer('beam_size', 3,
                        'The size of beam search for caption generation')




def main(_):
    config = Config()
    config.mode = FLAGS.mode
    config.train_cnn = FLAGS.train_cnn
    config.beam_size = FLAGS.beam_size
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    # 设置按需分配GPU
    with tf.Session(config=tf_config) as sess:
        if FLAGS.mode == 'train':
            # training mode
            data = prepare_train_data(config)
            model = CaptionGenerator(config)
            sess.run(tf.global_variables_initializer())
            if FLAGS.load:
                model.load(sess, FLAGS.model_file)
            if FLAGS.load_cnn:
                model.load_cnn(sess, FLAGS.cnn_model_file)
            tf.get_default_graph().finalize()
            model.train(sess, data)

        elif FLAGS.mode == 'eval':
            # evaluation mode
            coco, data, vocabulary = prepare_eval_data(config)
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            model.eval(sess, coco, data, vocabulary)

        else:
            # testing mode
            data, vocabulary = prepare_test_data(config)
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            model.test(sess, data, vocabulary)


if __name__ == '__main__':
    tf.app.run()
