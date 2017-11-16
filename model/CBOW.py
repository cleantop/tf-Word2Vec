import tensorflow as tf

class CBOW(object):
    def __init__(self, window_size, word_dimension, num_negative, num_vocab):

        # Context words
        self.input_x = tf.placeholder(tf.int32, [None, 2*window_size])

        # Target word
        self.input_y = tf.placeholder(tf.int32, [None, 1])

        # Valid words
        self.input_z = tf.placeholder(tf.int32, [1])

        with tf.device('/cpu:0'), tf.name_scope("word-embeddings"):
            self.W_in = tf.Variable(tf.random_uniform([num_vocab-1, word_dimension], maxval=1.0, minval=-1.0), name="W", trainable=True)
            zero_padding = tf.Variable(tf.zeros([1, word_dimension]), trainable=False)
            self.W_in = tf.concat([zero_padding, self.W_in], axis=0)

            self.projection = tf.nn.embedding_lookup(self.W_in, self.input_x)
            self.projection = tf.reduce_sum(self.projection, axis=1)


        with tf.device('/cpu:0'), tf.name_scope("negative-sampling"):
            self.W_out = tf.Variable(tf.random_uniform([num_vocab, word_dimension], maxval=1.0/50,minval=-1.0/50), name="W", trainable=True)
            b = tf.Variable(tf.constant(0.0, shape=[num_vocab], name="b"))

            self.loss = tf.nn.nce_loss(weights=self.W_out, biases=b, labels=self.input_y, inputs=self.projection, num_sampled=num_negative, num_classes=num_vocab)
            self.loss = tf.reduce_mean(self.loss)

        with tf.name_scope("near-words"):
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.W_in), 1, keep_dims=True))
            normalized_embeddings = self.W_in / norm
            valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, self.input_z)
            self.similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)