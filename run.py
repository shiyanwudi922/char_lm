#encoding=utf-8
import tensorflow as tf
import time
import math
import sys
import numpy as np

from read_utils import TextConverter, batch_generator, get_batch_cnt
from model import CharRNN
import os
import codecs

tf.app.flags.DEFINE_string('train_dir', 'default', 'directory for training')
tf.app.flags.DEFINE_string('model_name', 'default', 'name of the model')
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch size')
tf.app.flags.DEFINE_integer('max_time', 100, 'length of one training sample')
tf.app.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.app.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.app.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.app.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.app.flags.DEFINE_float('learning_rate', 0.5, 'initial learning rate')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5, 'Learning rate decays by this much.')
tf.app.flags.DEFINE_boolean('immediate_learning_rate_decay', False, 'decay learning rate immediately')
tf.app.flags.DEFINE_float("grad_clip", 5.0,"Clip gradients to this norm.")
tf.app.flags.DEFINE_float('keep_prob', 0.5, 'dropout rate during training')
tf.app.flags.DEFINE_string('input_file', '', 'utf8 encoded text file')
tf.app.flags.DEFINE_integer('max_train_steps', 100000, 'max steps to train')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 1000, 'How many training steps to do per checkpoint.')
tf.app.flags.DEFINE_integer('steps_per_log', 10, 'How many training steps to do per log.')
tf.app.flags.DEFINE_integer('max_vocab_size', 5000, 'max char number')
tf.app.flags.DEFINE_boolean("sampling", False, "Set to True for sampling.")
tf.app.flags.DEFINE_integer('sample_length', 30, 'max length to generate')

FLAGS = tf.app.flags.FLAGS

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

def create_model(session, num_classes, sampling, model_path):

    model = CharRNN(
        num_classes,
        FLAGS.batch_size,
        FLAGS.max_time,
        FLAGS.lstm_size,
        FLAGS.num_layers,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        FLAGS.grad_clip,
        sampling,
        FLAGS.keep_prob,
        FLAGS.use_embedding,
        FLAGS.embedding_size
    )
    # model_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
    # if(not os.path.exists(model_path)):
    #     os.mkdir(model_path)
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model

def train():

    with tf.Session() as sess:

        model_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
        checkpoint_path = os.path.join(model_path, "generate.ckpt")
        if(not os.path.exists(model_path)):
            os.makedirs(model_path)

        with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
            text = f.read().replace("\n", "")
        converter = TextConverter(text, FLAGS.max_vocab_size)
        print("actual vocabulary size is: " + str(converter.vocab_size))
        converter.save_to_file(os.path.join(model_path, 'converter.pkl'))

        arr = converter.text_to_arr(text)
        g = batch_generator(arr, FLAGS.batch_size, FLAGS.max_time)
        batch_cnt = get_batch_cnt(arr, FLAGS.batch_size, FLAGS.max_time)

        # create model
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.lstm_size))
        model = create_model(sess, converter.vocab_size, False, model_path)
        if(FLAGS.immediate_learning_rate_decay):
            sess.run(model.learning_rate_decay_op)

        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        initial_state = sess.run(model.initial_state)
        for inputs, targets in g:

            start_time = time.time()
            batch_loss, final_state = model.train_step(sess, inputs, targets, initial_state)
            step_time = (time.time()-start_time) / FLAGS.steps_per_checkpoint
            loss += batch_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            if current_step % batch_cnt == 0:
                initial_state = sess.run(model.initial_state)
                print("reset initial state")
            else:
                initial_state = final_state

            if current_step % FLAGS.steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print("global step %d learning rate %.4f step-time %.2f perplexity "
                      "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                step_time, perplexity))

                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)

                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0

                sys.stdout.flush()

            if current_step >= FLAGS.max_train_steps:
                break
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

def sample():

    with tf.Session() as sess:
        model_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
        converter = TextConverter(None, FLAGS.max_vocab_size, os.path.join(model_path, 'converter.pkl'))
        model = create_model(sess, converter.vocab_size, True, model_path)

        sys.stdout.write("> ")
        sys.stdout.flush()
        start_str = sys.stdin.readline().strip().decode('utf-8')
        while start_str:
            start = converter.text_to_arr(start_str)

            samples = [c for c in start]
            initial_state = sess.run(model.initial_state)
            x = np.zeros((1, 1))
            for c in start:
                x[0, 0] = c
                feed = {model.inputs: x,
                        model.initial_state: initial_state}
                preds, final_state = sess.run([model.proba_prediction, model.final_state],
                        feed_dict=feed)
                initial_state = final_state

            c = pick_top_n(preds, converter.vocab_size)
            while c == converter.vocab_size - 1:
                c = pick_top_n(preds, converter.vocab_size)
            samples.append(c)

            for i in range(FLAGS.sample_length):
                x[0, 0] = c
                feed = {model.inputs: x,
                        model.initial_state: initial_state}
                preds, final_state = sess.run([model.proba_prediction, model.final_state],
                                            feed_dict=feed)
                initial_state = final_state
                c = pick_top_n(preds, converter.vocab_size)
                while c == converter.vocab_size - 1:
                    c = pick_top_n(preds, converter.vocab_size)
                samples.append(c)

            print(converter.arr_to_text(np.array(samples)))

            sys.stdout.write("> ")
            sys.stdout.flush()
            start_str = sys.stdin.readline().strip().decode('utf-8')

        #     # 不断生成字符，直到达到指定数目
        #     for i in range(sample_length):
        #         x = np.zeros((1, 1))
        #         x[0, 0] = c
        #         feed = {self.inputs: x,
        #                 self.initial_state: new_state}
        #         preds, new_state = sess.run([self.proba_prediction, self.final_state],
        #                                     feed_dict=feed)
        #
        #         c = pick_top_n(preds, vocab_size)
        #         samples.append(c)
        #
        #     print(samples)


# def main(_):
#     model_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
#     if os.path.exists(model_path) is False:
#         os.makedirs(model_path)
#     with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
#         text = f.read().replace("\n", "")
#     converter = TextConverter(text, FLAGS.max_vocab)
#     converter.save_to_file(os.path.join(model_path, 'converter.pkl'))
#
#     arr = converter.text_to_arr(text)
#     g = batch_generator(arr, FLAGS.num_seqs, FLAGS.num_steps)
#     print(converter.vocab_size)
#     model = CharRNN(converter.vocab_size,
#                     num_seqs=FLAGS.num_seqs,
#                     num_steps=FLAGS.num_steps,
#                     lstm_size=FLAGS.lstm_size,
#                     num_layers=FLAGS.num_layers,
#                     learning_rate=FLAGS.learning_rate,
#                     train_keep_prob=FLAGS.train_keep_prob,
#                     use_embedding=FLAGS.use_embedding,
#                     embedding_size=FLAGS.embedding_size
#                     )
#     model.train(g,
#                 FLAGS.max_steps,
#                 model_path,
#                 FLAGS.save_every_n,
#                 FLAGS.log_every_n,
#                 )

def main(_):
    if FLAGS.sampling:
        sample()
    else:
        train()

if __name__ == '__main__':
    tf.app.run()
