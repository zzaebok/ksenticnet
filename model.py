import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn


def self_attention(inputs, name, return_alphas=False):
    shared = False
    # max_len = inputs.shape[1].value
    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer
    if shared:
        scope_name = 'self_attn'
    else:
        scope_name = 'self_attn' + name
    with tf.variable_scope(scope_name):
        den = True
        if den:
            x_proj = tf.layers.Dense(hidden_size)(inputs)
            x_proj = tf.nn.tanh(x_proj)
        else:
            x_proj = inputs
        u_w = tf.Variable(tf.random_normal([hidden_size, 1], stddev=0.01, seed=1227))
        # x_proj = tf.reshape(x_proj, shape=[-1, hidden_size])
        # x = tf.reshape(tf.matmul(x_proj, u_w), shape=[-1, max_len, 1])
        x = tf.tensordot(x_proj, u_w, axes=1)
        alphas = tf.nn.softmax(x, axis=-1)
        output = tf.matmul(tf.transpose(inputs, [0, 2, 1]), alphas)
        output = tf.squeeze(output, -1)
        if not return_alphas:
            return output
        else:
            return output, alphas


def ffn_layer(inputs, hidden_units, output_units, use_bias=True, bias_init=0., activation=tf.nn.tanh,
              scope="ffn_layer"):
    with tf.variable_scope(scope):
        dim = inputs.get_shape().as_list()[-1]
        # hidden layer
        hidden_weight = tf.get_variable(name="hidden_weight", shape=[dim, hidden_units], dtype=tf.float32)
        hidden_output = tf.matmul(inputs, hidden_weight)
        if use_bias:
            hidden_bias = tf.get_variable(name="hidden_bias", shape=[hidden_units], dtype=tf.float32,
                                          initializer=tf.constant_initializer(bias_init))
            hidden_output = tf.nn.bias_add(hidden_output, hidden_bias)
        hidden_output = activation(hidden_output)
        # output layer
        weight = tf.get_variable(name="weight", shape=[hidden_units, output_units], dtype=tf.float32)
        output = tf.matmul(hidden_output, weight)
        if use_bias:
            bias = tf.get_variable(name="bias", shape=[output_units], dtype=tf.float32,
                                   initializer=tf.constant_initializer(bias_init))
            output = tf.nn.bias_add(output, bias)
        output = activation(output)
        return output


class Model:
    def __init__(self, cfg, vocab_counts):
        # add data placeholders
        self.left_context = tf.placeholder(name="left_context", shape=[None, None], dtype=tf.int32)
        self.left_seq_len = tf.placeholder(name="left_seq_len", shape=[None], dtype=tf.int32)
        self.right_context = tf.placeholder(name="right_context", shape=[None, None], dtype=tf.int32)
        self.right_seq_len = tf.placeholder(name="right_seq_len", shape=[None], dtype=tf.int32)
        self.verb = tf.placeholder(name="verb", shape=[None], dtype=tf.int32)

        # add hyper-parameter placeholders
        self.batch_size = tf.placeholder(name="batch_size", dtype=tf.int32)
        self.is_train = tf.placeholder(name="is_train", shape=[], dtype=tf.bool)
        self.drop_rate = tf.placeholder(name="dropout_rate", dtype=tf.float32)
        self.lr = tf.placeholder(name="learning_rate", dtype=tf.float32)

        # build embedding lookup table
        with tf.device("/gpu:0"):
            with tf.variable_scope("context_lookup_table"):
                self.word_embeddings = tf.Variable(np.load(cfg.pretrained_context)["embeddings"],
                                                   name="word_embeddings",
                                                   dtype=tf.float32,
                                                   trainable=cfg.tune_emb)
                self.word_embeddings = tf.concat([tf.zeros([1, cfg.word_dim]), self.word_embeddings[1:, :]], axis=0)

            with tf.variable_scope("target_lookup_table"):
                self.verb_embeddings = tf.Variable(np.load(cfg.pretrained_target)["embeddings"],
                                                   name="verb_embeddings",
                                                   dtype=tf.float32,
                                                   trainable=cfg.tune_emb)
                #self.verb_embeddings = tf.concat([tf.zeros([1, cfg.word_dim]), self.verb_embeddings[1:, :]], axis=0)

            # negative sampling
            self.neg_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
                true_classes=tf.cast(tf.expand_dims(self.verb, axis=1), dtype=tf.int64),
                num_true=1,
                num_sampled=cfg.neg_sample,
                unique=True,
                range_max=cfg.verb_size,
                distortion=0.75,
                unigrams=vocab_counts))

            print('neg_ids : ', self.neg_ids)

        # embedding lookup
        # with tf.device("/gpu:0"):
            with tf.variable_scope("embedding_lookup"):
                left_context_emb = tf.nn.embedding_lookup(self.word_embeddings, self.left_context)
                right_context_emb = tf.nn.embedding_lookup(self.word_embeddings, self.right_context)
                verb_emb = tf.nn.embedding_lookup(self.verb_embeddings, self.verb)
                neg_verb_emb = tf.nn.embedding_lookup(self.verb_embeddings, self.neg_ids)

        # left context bi-lstm
        with tf.device("/gpu:0"):
            with tf.variable_scope("right_context_representation"):
                cell_fw = LSTMCell(num_units=cfg.num_units)
                cell_bw = LSTMCell(num_units=cfg.num_units)
                h_rc, _ = bidirectional_dynamic_rnn(cell_fw, cell_bw, right_context_emb,
                                                    sequence_length=self.right_seq_len,
                                                    dtype=tf.float32,
                                                    time_major=False,
                                                    scope="bi_lstm")
                h_rc = tf.concat(h_rc, axis=-1)
                # self-attention
                h_rc = self_attention(h_rc, name="self_attn_right")
                r_weight = tf.get_variable(name="r_weight",
                                           shape=[2 * cfg.num_units, 2 * cfg.num_units],
                                           dtype=tf.float32)
                h_rc = tf.nn.tanh(tf.matmul(h_rc, r_weight))
                print("right context shape: {}".format(h_rc.get_shape().as_list()))

            with tf.variable_scope("left_context_representation"):
                cell_fw = LSTMCell(num_units=cfg.num_units)
                cell_bw = LSTMCell(num_units=cfg.num_units)
                h_lc, _ = bidirectional_dynamic_rnn(cell_fw, cell_bw, left_context_emb,
                                                    sequence_length=self.left_seq_len,
                                                    dtype=tf.float32,
                                                    time_major=False,
                                                    scope="bi_lstm")

                h_lc = tf.concat(h_lc, axis=-1)  # shape = (batch_size, max_len, 2 * num_units)

                # self-attention
                h_lc = self_attention(h_lc, name="self_attn_left")  # shape = (batch_size, 2 * num_units)
                l_weight = tf.get_variable(name="l_weight",
                                           shape=[2 * cfg.num_units, 2 * cfg.num_units],
                                           dtype=tf.float32)
                h_lc = tf.nn.tanh(tf.matmul(h_lc, l_weight))

                print("left context shape: {}".format(h_lc.get_shape().as_list()))

            # with tf.device("/gpu:0"):

        with tf.device("/gpu:1"):
            with tf.variable_scope("neural_tensor_network"):
                T = tf.get_variable(name="T",
                                    shape=[cfg.output_units, 2 * cfg.num_units, 2 * cfg.num_units],
                                    dtype=tf.float32)
                W = tf.get_variable(name="W",
                                    shape=[4 * cfg.num_units, cfg.output_units],
                                    dtype=tf.float32)
                b = tf.get_variable(name="b",
                                    shape=[cfg.output_units],
                                    dtype=tf.float32)
                # compute tensors
                ff_product = tf.matmul(tf.concat([h_lc, h_rc], axis=-1), W)
                bilinear_list = []
                for k in range(cfg.output_units):
                    cur_res = tf.reduce_sum(tf.matmul(h_lc, T[k]) * h_rc, axis=1)
                    bilinear_list.append(cur_res)
                context = tf.nn.tanh(tf.reshape(tf.concat(bilinear_list, axis=0), shape=[-1, cfg.output_units]) +
                                     ff_product + b)  # shape = (batch_size, output_units)
                print("context representation shape: {}".format(context.get_shape().as_list()))

        # with tf.device("/gpu:1"):
            with tf.variable_scope("verb_representation"):
                target_verb = ffn_layer(verb_emb, cfg.num_units, cfg.output_units, scope="ffn_layer")
                print("verb representation shape: {}".format(target_verb.get_shape().as_list()))
                tf.get_variable_scope().reuse_variables()
                negative_verbs = ffn_layer(neg_verb_emb, cfg.num_units, cfg.output_units, scope="ffn_layer")
                print("negative verb shape: {}".format(negative_verbs.get_shape().as_list()))

            with tf.variable_scope("compute_loss"):
                true_logits = tf.reduce_sum(context * target_verb, axis=1)
                print("true logits shape: {}".format(true_logits.get_shape().as_list()))
                neg_logits = tf.matmul(context, tf.transpose(negative_verbs, [1, 0]))
                print("negative logits shape: {}".format(neg_logits.get_shape().as_list()))

        # with tf.device("/cpu:0"):

            with tf.variable_scope("nce_loss"):
                # cross-entropy(logits, labels)
                true_xent = tf.nn.sigmoid_cross_entropy_with_logits(logits=true_logits,
                                                                    labels=tf.ones_like(true_logits))
                sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(logits=neg_logits,
                                                                       labels=tf.zeros_like(neg_logits))

                # NCE-loss is the sum of the true and noise (sampled words) contributions, averaged over the batch.
                self.loss = (tf.reduce_sum(true_xent) + tf.reduce_sum(sampled_xent)) / tf.cast(self.batch_size,
                                                                                               dtype=tf.float32)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = optimizer.minimize(self.loss)

    def get_feed_dict(self, data, is_train=False, drop_rate=0.0, lr=None):
        feed_dict = {
            self.verb: data["vb"],
            self.left_context: data["lc"],
            self.left_seq_len: data["ll"],
            self.right_context: data["rc"],
            self.right_seq_len: data["rl"],
            self.batch_size: data["batch_size"]
        }
        if lr is not None:
            feed_dict[self.lr] = lr
        feed_dict[self.is_train] = is_train
        feed_dict[self.drop_rate] = drop_rate
        return feed_dict