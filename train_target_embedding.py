import tensorflow as tf
import os
import numpy as np
from data_helper import read_vocab_to_dict, build_fasttext, dataset_iterator, load_verb_count
from model import Model
import pickle

if __name__ == "__main__":


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # read word and verb dict
    print("load dictionary...")
    example_dict = dict()
    with open('vocabulary/example_dict.pkl', 'rb') as f:
        example_dict = pickle.load(f)
    word_dict = read_vocab_to_dict("vocabulary/word_vocab.txt")
    verb_dict = read_vocab_to_dict("vocabulary/verb_vocab.txt")
    verb_vocab_count = load_verb_count("vocabulary/verb_count.txt")

    flags = tf.flags
    flags.DEFINE_integer("neg_sample", 10, "number of negative samples")
    flags.DEFINE_integer("word_dim", 300, "word embedding dimension")
    flags.DEFINE_integer("num_units", 100, "number of units for rnn cell and hidden layer of ffn")
    flags.DEFINE_integer("output_units", 200, "number of units for output part")
    flags.DEFINE_bool("use_pretrained", True, "use pretrained word2vec")
    flags.DEFINE_bool("tune_emb", True, "tune pretrained embeddings while training")
    flags.DEFINE_string("pretrained_context", "embeddings/context_embeddings.npz", "pretrained context embedding path")
    flags.DEFINE_string("pretrained_target", "embeddings/target_embeddings.npz", "pretrained target embedding path")
    flags.DEFINE_integer("vocab_size", len(word_dict), "word vocab size")
    flags.DEFINE_integer("verb_size", len(verb_dict), "verb vocab size")
    flags.DEFINE_float("lr", 0.001, "learning_rate")
    flags.DEFINE_integer("batch_size", 300, "batch size")
    flags.DEFINE_integer("epochs", 3, "epochs")
    flags.DEFINE_string("ckpt", "ckpt/", "checkpoint path")
    flags.DEFINE_string("model_name", "train_concept", "model name")
    config = flags.FLAGS


    # initialize with pretrained fasttext embeddings
    if not os.path.exists(config.pretrained_context) or not os.path.exists(config.pretrained_target):
        build_fasttext(config.fasttext_path, config.pretrained_context, config.pretrained_target, word_dict, verb_dict, config.word_dim)

    if not os.path.exists(config.ckpt):
        os.makedirs(config.ckpt)

    # training the model
    print("start training...")

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        # build model
        print("build model...")
        model = Model(config, verb_vocab_count)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1)
        for epoch in range(config.epochs):
            print('epoch :', epoch)
            for i, data in enumerate(dataset_iterator(example_dict, word_dict, verb_dict, config.batch_size)):
                feed_dict = model.get_feed_dict(data, is_train=True, lr=config.lr)
                _, losses = sess.run([model.train_op, model.loss], feed_dict=feed_dict)
        # save the model
        saver.save(sess, config.ckpt + config.model_name, global_step=config.epochs)
        # save the trained target embedding
        target_emb = sess.run(model.verb_embeddings)
        np.savez_compressed("embeddings/trained_target_emb.npz", embeddings=target_emb)
