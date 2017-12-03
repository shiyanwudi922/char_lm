# coding: utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os



class CharRNN:
    def __init__(self,
                 num_classes,
                 batch_size=64,
                 # max_time=50,
                 lstm_size=128,
                 num_layers=2,
                 learning_rate=0.01,
                 learning_rate_decay_factor = 0.8,
                 grad_clip=5,
                 sampling=False,
                 keep_prob=0.5,
                 use_embedding=False,
                 embedding_size=128,
                 use_sample_loss=False):
        if sampling is True:
            # self.batch_size, self.max_time = 1, 1
            self.batch_size = 1
        else:
            # self.batch_size, self.max_time = batch_size, max_time
            self.batch_size = batch_size

        self.num_classes = num_classes
        # self.batch_size = batch_size
        # self.max_time = max_time
        self.lstm_size = lstm_size
        self.num_layers = num_layers

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        self.grad_clip = grad_clip
        self.sampling = sampling
        self.keep_prob = keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size
        self.use_sample_loss = use_sample_loss

        # tf.reset_default_graph()
        self.build_inputs()
        self.build_lstm()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()

    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=(
                self.batch_size, None), name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=(
                self.batch_size, None), name='targets')
            # self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            # 对于中文，需要使用embedding层
            # 英文字母没有必要用embedding层
            if self.use_embedding is False:
                self.lstm_inputs = tf.one_hot(self.inputs, self.num_classes)
            else:
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable('embedding', [self.num_classes, self.embedding_size])
                    self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)

    def build_lstm(self):
        # 创建单个cell并堆叠多层
        # def get_a_cell(lstm_size, keep_prob):
        #     cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
        #     if(not self.sampling):
        #         cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        #     return cell

        with tf.name_scope('lstm'):

            def single_cell():
                return tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)
            if not self.sampling:
                def single_cell():
                    return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size),
                                                         output_keep_prob=self.keep_prob)
            cell = single_cell()
            if self.num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([single_cell() for _ in range(self.num_layers)])

            self.initial_state = cell.zero_state(self.batch_size, tf.float32)

            # 通过dynamic_rnn对cell展开时间维度
            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.lstm_inputs, initial_state=self.initial_state)

            # 通过lstm_outputs得到概率
            seq_output = tf.concat(self.lstm_outputs, 1)
            self.x = tf.reshape(seq_output, [-1, self.lstm_size])

            with tf.variable_scope('softmax'):
                self.softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes], stddev=0.1))
                self.softmax_b = tf.Variable(tf.zeros(self.num_classes))

            self.logits = tf.matmul(self.x, self.softmax_w) + self.softmax_b
            self.proba_prediction = tf.nn.softmax(self.logits, name='predictions')

    def build_loss(self):
        with tf.name_scope('loss'):
            def sampled_loss(labels, inputs, w, b, num_classes):
                local_labels = tf.reshape(labels, [-1, 1])
                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                local_w_t = tf.cast(tf.transpose(w), tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(inputs, tf.float32)
                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        weights=local_w_t,
                        biases=local_b,
                        labels=local_labels,
                        inputs=local_inputs,
                        num_sampled=512,
                        num_classes=num_classes),
                    tf.float32
                )

            if(self.use_sample_loss):
                self.loss = tf.reduce_mean(sampled_loss(self.targets, self.x, self.softmax_w, self.softmax_b, self.num_classes))
            else:
                y_one_hot = tf.one_hot(self.targets, self.num_classes)
                y_reshaped = tf.reshape(y_one_hot, [-1, self.num_classes])
                loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
                self.loss = tf.reduce_mean(loss)

    def build_optimizer(self):
        # 使用clipping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

    def train_step(self, sess, inputs, targets, initial_state):

        feed = {self.inputs: inputs,
                self.targets: targets,
                self.initial_state: initial_state}
        batch_loss, final_state, _ = sess.run([self.loss, self.final_state, self.train_op],
                                              feed_dict=feed)

        return batch_loss, final_state

    def set_learning_rate(self, sess, learning_rate):
        sess.run(self.learning_rate.assign(learning_rate))

