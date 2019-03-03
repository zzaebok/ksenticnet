import os
import codecs
import math
import random
import numpy as np
from collections import Counter
import gensim
import pickle
from konlpy import jvm
from gensim.models import FastText
from konlpy.tag import Kkma
import jpype
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

PAD = "<pad>"
UNK = "<unk>"


def build_fasttext(filename, context_path, target_path, word_dict, verb_dict, dim):
    scale = np.sqrt(3.0 / dim)
    context_emb = np.random.uniform(-scale, scale, [len(word_dict), dim])
    target_emb = np.random.uniform(-scale, scale, [len(verb_dict), dim])
    fasttext_model = FastText.load_fasttext_format(filename, encoding='utf8')

    for word in word_dict:
        idx = word_dict[word]
        context_emb[idx] = fasttext_model.wv[word]

    for word in verb_dict:
        idx = verb_dict[word]
        target_emb[idx] = fasttext_model.wv[word]

    np.savez_compressed(context_path, embeddings=context_emb)
    np.savez_compressed(target_path, embeddings=target_emb)

def build_vocab(example_dict_pkl, word_threshold=5):
    tokenizer = Kkma()
    word_counter, verb_counter = Counter(), Counter()
    example_dict = dict()
    with open(example_dict_pkl, 'rb') as f:
        example_dict = pickle.load(f)

    for word, example in example_dict.items():
        sentence = example[0] + word + example[1]
        verb_counter[word] += 1
        for x in tokenizer.pos(sentence):
            if x[0] in example_dict.keys():
                verb_counter[x[0]] += 1
            else:
                word_counter[x[0]] += 1

    # build word vocabulary
    word_vocab = [PAD] + [word for word, count in word_counter.most_common() if count >= word_threshold] + [UNK]
    verb_vocab = [word for word, count in verb_counter.most_common()]

    # save to file
    with open("vocabulary/word_vocab.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(word_vocab))
    with open("vocabulary/verb_vocab.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(verb_vocab))
    with open("vocabulary/verb_count.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(["{}\t{}".format(word, count) for word, count in verb_counter.most_common()]))

def load_verb_count(filename):
    count_list = []
    with codecs.open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split("\t")
            count = int(line[1])
            count_list.append(count)
    return count_list

def read_vocab_to_dict(filename):
    vocab = dict()
    with open(filename, mode="r", encoding="utf-8") as f:
        for idx, word in enumerate(f):
            word = word.lstrip().rstrip()
            vocab[word] = idx
    #return word_dict
    return vocab

def pad_sequence(sequence, pad_tok=0, max_length=None):
    """Pad batched dataset with shape = (batch_size, seq_length(various))
    :param sequence: input sequence
    :param pad_tok: padding token, default is 0
    :param max_length: max length of padded sequence, default is None
    :return: padded sequence
    """
    if max_length is None:
        max_length = max([len(seq) for seq in sequence])
    sequence_padded, seq_length = [], []
    for seq in sequence:
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded.append(seq_)
        seq_length.append(min(len(seq), max_length))
    return sequence_padded, seq_length

def build_batch_dataset(left_context, verbs, right_context):
    left_context, left_seq_len = pad_sequence(left_context)
    right_context, right_seq_len = pad_sequence(right_context)
    batch_size = len(verbs)
    return {"lc": left_context, "ll": left_seq_len, "rc": right_context, "rl": right_seq_len, "vb": verbs,
            "batch_size": batch_size}

def dataset_iterator(example_dict, word_dict, verb_dict, batch_size):
    tokenizer = Kkma()
    left_context, verbs, right_context = [], [], []
    for word, example in example_dict.items():
        # split data
        l_c, vb, r_c = example[0], word, example[1]
        # convert to indices
        l_c = [word_dict[word[0]] if word[0] in word_dict else word_dict[UNK] for word in tokenizer.pos(l_c)]
        vb = verb_dict[vb]
        r_c = [word_dict[word[0]] if word[0] in word_dict else word_dict[UNK] for word in tokenizer.pos(r_c)]
        # add to list
        left_context.append(l_c)
        verbs.append(vb)
        right_context.append(r_c)
        # yield batched dataset
        if len(left_context) == batch_size:
            yield build_batch_dataset(left_context, verbs, right_context)
            left_context, verbs, right_context = [], [], []
    if len(left_context) > 0:
        yield build_batch_dataset(left_context, verbs, right_context)


if __name__ == "__main__":
    build_vocab('vocabulary/example_dict.pkl')
    word_dict = read_vocab_to_dict("vocabulary/word_vocab.txt")
    verb_dict = read_vocab_to_dict('vocabulary/verb_vocab.txt')
    verb_vocab_count = load_verb_count("vocabulary/verb_count.txt")
    build_fasttext('wiki.ko.bin', 'embeddings/context_embeddings.npz', 'embeddings/target_embeddings.npz', word_dict, verb_dict, 300)