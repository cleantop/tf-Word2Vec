import tensorflow as tf
import collections
import os
import time
import datetime
import numpy as np

from sklearn.utils import shuffle
from utills import utill
from model.Skipgram import Skipgram


word_dimension = 100
num_max_word = 50000
num_epoch = 5
num_negative = 25
window_size = 1
batch_size = 100

directory = "./skipgram_model_log"
if not os.path.exists(directory):
    os.makedirs(directory)
model_path = directory+"/model.ckpt"


data = utill.load_data("./dataset/text8")
word_counter = collections.Counter(data)
data, word_vocab, iword_vocab = utill.build_train_data(data, word_counter, num_max_word)

def get_next_batch():
    global data_pointer
    global data

    batch_x = []
    batch_y = []

    while len(batch_x) <= batch_size:
        if len(data) <= data_pointer:
            break

        target = data[data_pointer]
        if target == 0:# UNK Token
            data_pointer += 1
            continue
        for j in range(-window_size, window_size + 1):
            word_index = data_pointer + j

            if word_index < 0 or word_index >= len(data) or j == 0:
                continue

            if data[word_index] == 0:# UNK Token
                data_pointer += 1
                break

            batch_x.append([target])
            batch_y.append([data[word_index]])

            data_pointer += 1
    return batch_x, batch_y


def norm_mat(a):
    row_sums = a.sum(axis=1)
    new_matrix = a / row_sums[:, np.newaxis]
    return new_matrix

def norm_vec(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

with tf.Graph().as_default():
    with tf.Session() as sess:
        with sess.as_default():

            # window_size, word_dimension, num_negative, num_vocab
            skipgram = Skipgram(
                window_size=window_size,
                word_dimension=word_dimension,
                num_negative=num_negative,
                num_vocab=len(word_vocab)
            )

            saver = tf.train.Saver()

            # Define Training procedures
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
            grads_and_vars = optimizer.compute_gradients(skipgram.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            loss_summary = tf.summary.scalar("loss", skipgram.loss)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "skipgram_runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                '''
                    cnn training step
                '''
                feed_dict = {
                    skipgram.input_x: x_batch,
                    skipgram.input_y: y_batch
                }

                _, step, summaries, loss = sess.run([train_op, global_step, train_summary_op, skipgram.loss], feed_dict=feed_dict)
                train_summary_writer.add_summary(summaries, step)
                return loss

            def valid_step():
                feed_dict = {
                    skipgram.input_x: [[0]],
                    skipgram.input_y: [[0]],
                    skipgram.input_z: [word_vocab["three"]]
                }

                e = sess.run(skipgram.similarity, feed_dict=feed_dict)
                nearest = e[0].argsort()[-10:][::-1]
                for word in nearest:
                    print(iword_vocab[word], end=" ")
                print()

            for step in range(num_epoch):
                data_pointer = 0
                avg_loss = 0
                num_iter = 0
                while data_pointer < len(data):
                    batch_x, batch_y = get_next_batch()
                    loss = train_step(batch_x, batch_y)
                    avg_loss += loss
                    num_iter += 1

                    if num_iter % 1000 == 0:
                        valid_step()
                        print("step {}, loss {:g}".format(num_iter, avg_loss/(num_iter*batch_size)))

                avg_loss /= (batch_size * num_iter)
                time_str = datetime.datetime.now().isoformat()
                print("{}: epoch {}, loss {:g}".format(time_str, step, avg_loss))
            save_path = saver.save(sess, model_path)