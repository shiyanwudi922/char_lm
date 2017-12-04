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
tf.app.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.app.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.app.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.app.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.app.flags.DEFINE_boolean("use_sample_loss", False, "whether to use candidate sampling.")
tf.app.flags.DEFINE_float('learning_rate', 0.1, 'initial learning rate')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5, 'Learning rate decays by this much.')
tf.app.flags.DEFINE_float('set_learning_rate', 0.0, 'set learning rate by hand')
tf.app.flags.DEFINE_float("grad_clip", 5.0,"Clip gradients to this norm.")
tf.app.flags.DEFINE_float('keep_prob', 0.5, 'dropout rate during training')
tf.app.flags.DEFINE_string('input_file', '', 'utf8 encoded text file')
tf.app.flags.DEFINE_integer('max_train_steps', 10000, 'max steps to train')
tf.app.flags.DEFINE_integer('steps_per_sentence_length', 1000, 'max steps to train')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 100, 'How many training steps to do per checkpoint.')
tf.app.flags.DEFINE_integer('steps_per_log', 100, 'How many training steps to print info.')
tf.app.flags.DEFINE_integer('max_vocab_size', 5000, 'max char number')
tf.app.flags.DEFINE_boolean("sampling", False, "Set to True for sampling.")
tf.app.flags.DEFINE_integer('sample_length', 30, 'max length to generate')

FLAGS = tf.app.flags.FLAGS

train_sentence_length = [10, 30, 50, 70, 100, 150]

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
        # max_time,
        FLAGS.lstm_size,
        FLAGS.num_layers,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        FLAGS.grad_clip,
        sampling,
        FLAGS.keep_prob,
        FLAGS.use_embedding,
        FLAGS.embedding_size,
        FLAGS.use_sample_loss
    )
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
        if (not os.path.exists(model_path)):
            os.makedirs(model_path)
        checkpoint_path = os.path.join(model_path, "generate.ckpt")

        with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
            text = f.read() #.replace("\n", "")
        converter_path = os.path.join(model_path, 'converter.pkl')
        if( not os.path.exists(converter_path) ):
            print("construct converter.")
            converter = TextConverter(text, FLAGS.max_vocab_size)
            converter.save_to_file(os.path.join(model_path, 'converter.pkl'))
        else:
            print("load converter")
            converter = TextConverter(None, FLAGS.max_vocab_size, converter_path)
        print("actual vocabulary size is: " + str(converter.vocab_size))

        arr = converter.text_to_arr(text)
        sent_len_p = [1.0/len(train_sentence_length) for l in train_sentence_length]
        max_time = np.random.choice(train_sentence_length, 1, p=sent_len_p)[0]
        batch_cnt = get_batch_cnt(arr, FLAGS.batch_size, max_time)
        current_step_batch = 0

        # create model
        print("Creating %d layers of %d units for max time %d." % (FLAGS.num_layers, FLAGS.lstm_size, max_time))
        model = create_model(sess, converter.vocab_size, False, model_path)
        if(FLAGS.set_learning_rate > 0):
            model.set_learning_rate(sess, FLAGS.set_learning_rate)

        loss_per_checkpoint = 0.0
        current_step = 0
        previous_losses = []
        initial_state = sess.run(model.initial_state)
        while True:
            g = batch_generator(arr, FLAGS.batch_size, max_time)
            for inputs, targets in g:

                start_time = time.time()
                batch_loss, final_state = model.train_step(sess, inputs, targets, initial_state)
                step_time = time.time() - start_time
                loss_per_checkpoint += batch_loss / FLAGS.steps_per_checkpoint
                current_step += 1
                current_step_batch += 1

                if current_step % FLAGS.steps_per_log == 0:
                    perplexity = math.exp(float(batch_loss)) if batch_loss < 300 else float("inf")
                    print("global step %d learning rate %.4f step-time %.2f perplexity "
                          "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                    step_time, perplexity))

                if current_step % FLAGS.steps_per_checkpoint == 0:
                    if len(previous_losses) > 2 and loss_per_checkpoint > max(previous_losses[-3:]) and sess.run(model.learning_rate) >= 0.0002:
                        sess.run(model.learning_rate_decay_op)
                    previous_losses.append(loss_per_checkpoint)
                    loss_per_checkpoint = 0.0
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                if current_step_batch % batch_cnt == 0:
                    print("reset initial state")
                    initial_state = sess.run(model.initial_state)
                    current_step_batch = 0
                else:
                    initial_state = final_state

                if current_step % FLAGS.steps_per_sentence_length == 0:
                    max_time = np.random.choice(train_sentence_length, 1, p=sent_len_p)[0]
                    print("change max time: %d" % (max_time))
                    batch_cnt = get_batch_cnt(arr, FLAGS.batch_size, max_time)
                    current_step_batch = 0
                    initial_state = sess.run(model.initial_state)
                    break

                if current_step >= FLAGS.max_train_steps:
                    break

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
        start_str = sys.stdin.readline().decode('utf-8')
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
            start_str = sys.stdin.readline().decode('utf-8')

def main(_):
    if FLAGS.sampling:
        sample()
    else:
        train()

if __name__ == '__main__':
    tf.app.run()
