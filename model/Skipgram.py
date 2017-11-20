import tensorflow as tf

class Skipgram(object):
    def __init__(self, window_size, word_dimension, num_negative, num_vocab):

        # Target word
        self.input_x = tf.placeholder(tf.int32, [None, 1])

        # context word
        self.input_y = tf.placeholder(tf.int32, [None, 1])

        # Valud words
        self.input_z = tf.placeholder(tf.int32, [1])

        with tf.device('/cpu:0'), tf.name_scope("word-embeddings"):
            self.W_in = tf.Variable(tf.random_uniform([num_vocab, word_dimension], maxval=1.0, minval=-1.0), name="input_embedding",
                                     trainable=True)
            self.projection = tf.nn.embedding_lookup(self.W_in, self.input_x)  # None, 2*window_size, word_dimension
            self.projection = tf.reshape(self.projection, [-1, word_dimension])


        with tf.device('/cpu:0'), tf.name_scope("negative-sampling"):
            self.W_out = tf.Variable(tf.random_uniform([num_vocab, word_dimension], maxval=1.0/word_dimension, minval=-1.0/word_dimension), name="output_embedding",
                                     trainable=True)
            b = tf.Variable(tf.constant(0.0, shape=[num_vocab], name="b"))

            # Negative sampling
            self.loss = tf.nn.nce_loss(weights=self.W_out, biases=b, labels=self.input_y, inputs=self.projection,
                                       num_sampled=num_negative, num_classes=num_vocab)
            self.loss = tf.reduce_mean(self.loss)

        with tf.name_scope("near-words"):
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.W_in), 1, keep_dims=True))
            self.normalized_embeddings = self.W_in / norm
            valid_embeddings = tf.nn.embedding_lookup(
                self.normalized_embeddings, self.input_z)
            self.similarity = tf.matmul(valid_embeddings, self.normalized_embeddings, transpose_b=True)