import tensorflow as tf
import collections
import datetime
import numpy as np

from sklearn.utils import shuffle
from utills import utill
from model.CBOW import CBOW

word_dimension = 100
num_max_word = 50000
num_epoch = 5
num_negative = 25
window_size = 3
batch_size = 100

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

        context = []
        for j in range(-window_size, window_size + 1):

            word_index = data_pointer + j
            if word_index < 0 or word_index >= len(data) or j == 0:
                continue
            if data[word_index] == 0:# UNK Token
                data_pointer += 1
                break
            context.append(data[word_index])

        if len(context) != 2*window_size:
            data_pointer += 1
            continue
        batch_x.append(context)
        batch_y.append([target])

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
            cbow = CBOW(
                window_size=window_size,
                word_dimension=word_dimension,
                num_negative=num_negative,
                num_vocab=len(word_vocab)
            )

            saver = tf.train.Saver()

            # Define Training procedures
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
            grads_and_vars = optimizer.compute_gradients(cbow.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                '''
                    cnn training step
                '''
                feed_dict = {
                    cbow.input_x: x_batch,
                    cbow.input_y: y_batch
                }

                _, step, loss = sess.run([train_op, global_step, cbow.loss], feed_dict=feed_dict)
                return loss

            def valid_step():
                feed_dict = {
                    cbow.input_x: [[0]*(2*window_size)],
                    cbow.input_y: [[0]],
                    cbow.input_z: [word_vocab["three"]]
                }

                e = sess.run(cbow.similarity, feed_dict=feed_dict)
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