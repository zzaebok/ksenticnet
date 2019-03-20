from senticnet5 import senticnet
import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import numpy as np
import math
import pickle
from wordfreq import word_frequency
nltk.download('wordnet')

##### basic functions #####
class Hypernyms:
    def __init__(self, word, synset_obj):
        self.level = [[],[],[],[]]
        self.level[0] = [lemma.name().lower() for lemma in synset_obj.lemmas()]
        self.level[0].remove(word)
        l1, l2, l3 = [], [], []
        l1 = synset_obj.hypernyms()
        for hyper in l1:
            self.level[1] += [lemma.name().lower() for lemma in hyper.lemmas()]
            l2 += hyper.hypernyms()
        for hyper in l2:
            self.level[2] += [lemma.name().lower() for lemma in hyper.lemmas()]
            l3 += hyper.hypernyms()
        for hyper in l3:
            self.level[3] += [lemma.name().lower() for lemma in hyper.lemmas()]

def weighted_sentic(sv1, fr1, sv2, fr2):
    #input: sentic values and their frequencies. 0 <= frequency <= 1
    #output: weighted sum of sentic values

    #polarity values must be equal
    polarity = sv1[6]

    tf = fr1+fr2
    fr1 /= tf
    fr2 /= tf

    pl = float(sv1[0]) * fr1 + float(sv2[0]) * fr2
    at = float(sv1[1]) * fr1 + float(sv2[1]) * fr2
    se = float(sv1[2]) * fr1 + float(sv2[2]) * fr2
    ap = float(sv1[3]) * fr1 + float(sv2[3]) * fr2
    pv = float(sv1[7]) * fr1 + float(sv2[7]) * fr2

    sentic = [pl, at, se, ap]

    return [round(pl, 3), round(at, 3), round(se, 3), round(ap, 3)] + sentic_to_sentiments(sentic) + [polarity, round(pv, 3)]

def sentic_to_sentiments(sentic):
    #input: sentic values only [0.1, 0.2, 0.3, 0.4]
    #output: sentiments #joy, #admiration

    sentiment_dict = {
            # col 1  or   col 2
        0: ['#sadness', '#joy'],
        1: ['#surprise', '#interest'],
        2: ['#fear', '#anger'],
        3: ['#disgust', '#admiration']
    }

    first_sentic, second_sentic = np.argsort(np.absolute(sentic))[-2:][::-1]
    col1 = math.floor(sentic[first_sentic]) + 1
    col2 = math.floor(sentic[second_sentic]) + 1

    return[sentiment_dict[first_sentic][col1], sentiment_dict[second_sentic][col2]]

def sentic_to_string(sentic_value):
    #input: sentic value
    #output: its string it is for recording in txt file
    string = ''
    for s in sentic_value:
        string += '\''+str(s)+'\', '
    return string[:-2]

def load_korean_wordnet_offset(kw_path):
    kw_offset = []
    with open(kw_path, 'r', encoding='utf8') as f:
        for line in f.read().split('\n')[1:]:
            kw_offset.append(line.split('\t')[0])
    return kw_offset

def load_similarities(trained_emb_path):
    trained_target_emb = np.load(open(trained_emb_path, 'rb'))['embeddings']
    _sparse = sparse.csr_matrix(trained_target_emb)
    similarities = cosine_similarity(_sparse)
    return similarities

def read_vocab_to_dict(filename):
    vocab = dict()
    with open(filename, mode="r", encoding="utf-8") as f:
        for idx, word in enumerate(f):
            word = word.lstrip().rstrip()
            vocab[word] = idx
    #return word_dict
    return vocab

def index_to_word(word_dict):
    idx_word = dict()
    for word, idx in word_dict.items():
        idx_word[idx] = word
    return idx_word


##### direct mapping #####

def direct_mapping(korean_wordnet_path):
    print('Direct mapping started...')
    # input: korean_wordnet_path - 00018158-v	Verb	uprise, rise, arise, turn_out, get_up	기상하
    # output: offset_sentic_dict["00044455-n"] = ['emergence', '#joy', '#surprise', 'positive', '0.726', 'appearance', 'start', 'casus_belli', 'beginning', 'egress']

    # step 1: make word - synset dict
    word_synset_dict = dict()
    with open(korean_wordnet_path, 'r', encoding='utf8') as f:
        lines = f.read().split('\n')[1:]
        for line in lines:
            words = line.split('\t')[2].replace(' ','').split(',')
            offset = line.split('\t')[0]
            for word in words:
                if word not in word_synset_dict:
                    word_synset_dict[word] = [offset]
                else:
                    word_synset_dict[word].append(offset)


    # step 2: make synset - sentic dict
    with open('vocabulary/affectnet_dict.pkl', 'rb') as f:
        aff_fr_dict = pickle.load(f)
    fr_offset_dict = dict()
    offset_sentic_dict = dict()
    deleted_offset = []
    for word in word_synset_dict:
        if word not in senticnet:
            continue

        found = False
        synsets = [wn.synset_from_pos_and_offset(synset_id[-1], int(synset_id[:8].lstrip('0'))) for synset_id in word_synset_dict[word]]
        num_synsets = len(synsets)
        hypernyms = [Hypernyms(word, synset) for synset in synsets]
        for i in range(4):
            for j in range(num_synsets):
                offset = word_synset_dict[word][j]
                if offset in deleted_offset:
                    continue

                for hypernym in hypernyms[j].level[i]:
                    if hypernym not in senticnet[word]:
                        continue

                    # hypernym is in semantics so, the synset could represent the meaning of the word
                    if offset not in offset_sentic_dict:
                        offset_sentic_dict[offset] = senticnet[word]
                        fr_offset_dict[offset] = aff_fr_dict[word]
                    else:
                        # if polarity is different, deleted existing offset_sentic pair.
                        if offset_sentic_dict[offset][6] != senticnet[word][6]:
                            del offset_sentic_dict[offset]
                            deleted_offset.append(offset)
                            #the offset is not allowed, so we no longer need to see other hypernyms
                            break

                        offset_sentic_dict[offset] = weighted_sentic(offset_sentic_dict[offset], fr_offset_dict[offset], senticnet[word], aff_fr_dict[word])
                        fr_offset_dict[offset] += aff_fr_dict[word]
                    found = True
                    break
            # no deeper progress
            if found:
                # lesk algorithm is applied to those which were not mapped directly ( for later use )
                del senticnet[word]
                break

    return offset_sentic_dict

##### lesk algorithm #####

def apply_lesk(offset_sentic_dict):
    print('Appling Lesk Algorithms Started...')
    # making another offset_sentic_dict for rest of concepts and concat it to the existed one.
    # input: offset_sentic_dict["00044455-n"] = ['0.1', '0.1', '0.1', '0.1', #joy', '#surprise', 'positive', '0.726', 'appearance', 'start', 'casus_belli', 'beginning', 'egress'] // semantics might not included
    # output: last_offset_sentic_dict["00044455-n"] = ['0.1', '0.1', '0.1', '0.1', '#joy', '#surprise', 'positive', '0.726', 'appearance', 'start', 'casus_belli', 'beginning', 'egress']

    with open('vocabulary/affectnet_dict.pkl', 'rb') as f:
        aff_fr_dict = pickle.load(f)
    fr_offset_dict = dict()

    deleted_offset = []

    #direct mapped words were deleted before
    for word, value in senticnet.items():
        context = word
        found = False
        for i in range(8, 13):
            context += senticnet[word][i] + ' '
        try:
            synset = lesk(context, word)
            offset = str(synset.offset()).zfill(8) + '-' + synset.pos()
            found = True
        except AttributeError:
            # not found
            # sequentially, because it is arranged by c. similarity
            for v in value[8:13]:
                try:
                    synset = lesk(context, v)
                    offset = str(synset.offset()).zfill(8) + '-' + synset.pos()
                    found = True
                except AttributeError:
                    continue
        if found == False:
            continue
        # Direct mapped offset is not considered
        if offset in offset_sentic_dict:
            continue
        if offset in deleted_offset:
            continue

        if offset not in offset_sentic_dict:
            offset_sentic_dict[offset] = value
            fr_offset_dict[offset] = aff_fr_dict[word]
        else:
            if offset_sentic_dict[offset][6] != value[6]:
                del offset_sentic_dict[offset]
                deleted_offset.append(offset)
                continue
            else:
                offset_sentic_dict[offset] = weighted_sentic(offset_sentic_dict[offset], fr_offset_dict[offset], value, aff_fr_dict[word])
                fr_offset_dict[offset] += aff_fr_dict[word]

    return offset_sentic_dict

##### main #####

if __name__ == "__main__":

    ksenticnet = dict()
    similarity_dict = dict()    # to store max similarity on each synset for a korean word
    similarity_matrix = load_similarities('embeddings/trained_target_emb.npz')
    word_idx = read_vocab_to_dict('vocabulary/verb_vocab.txt')
    idx_word = index_to_word(word_idx)
    korean_wordnet_path = 'kwn_1.0/kwn_synset_list.tsv'

    offset_sentic_dict = direct_mapping(korean_wordnet_path)
    offset_sentic_dict = apply_lesk(offset_sentic_dict)

    fr_offset_dict = dict()
    fr_ko_dict = dict()

    deleted_kor = []

    lines = open(korean_wordnet_path, 'r', encoding='utf8').read().split('\n')[1:]

    print('Making KSenticnet Started...')
    for line in lines:
        offset = line.split('\t')[0]

        if offset not in offset_sentic_dict:
            continue

        en_words = line.split('\t')[2].replace(' ','').split(',')
        kor_words = line.split('\t')[3].replace(' ', '').split(',')

        offset_freq = np.sum([word_frequency(en_word.replace('_', ' '), 'en') for en_word in en_words])
        fr_offset_dict[offset] = offset_freq

        avg_similarity = lambda kor, kors: np.sum([similarity_matrix[word_idx[kor]][word_idx[x]] for x in kors]) / len(kors)

        for kor_word in kor_words:
            if kor_word in deleted_kor:
                continue
            ksemantics = kor_words[:]
            ksemantics.remove(kor_word)
            if kor_word not in ksenticnet:
                #only one korean word in one synset
                if len(kor_words) == 1:
                    similarity_dict[kor_word] = 1
                    fr_ko_dict[kor_word] = fr_offset_dict[offset]
                else:
                    similarity_dict[kor_word] = avg_similarity(kor_word, ksemantics)
                ksenticnet[kor_word] = offset_sentic_dict[offset][:8] + ksemantics
            else:
                if len(kor_words) == 1:
                    # existed one was not the only word for synset. Which means that this one has to represent the value.
                    if similarity_dict[kor_word] != 1:
                        similarity_dict[kor_word] = 1
                        fr_ko_dict[kor_word] = fr_offset_dict[offset]
                        ksenticnet[kor_word] = offset_sentic_dict[offset][:8]
                    else:
                        if ksenticnet[kor_word][6] != offset_sentic_dict[offset][6]:
                            del ksenticnet[kor_word]
                            deleted_kor.append(kor_word)
                            continue
                        fr_ko_dict[kor_word] += fr_offset_dict[offset]
                        # to prevent both frequencies are 0
                        ksenticnet[kor_word] = weighted_sentic(ksenticnet[kor_word], max(fr_ko_dict[kor_word], 1), offset_sentic_dict[offset], max(fr_offset_dict[offset], 1))
                #we have to compare each synset's similarity on the korean word and choose only one.
                else:
                    avg_simil = avg_similarity(kor_word, ksemantics)
                    if avg_simil > similarity_dict[kor_word]:
                        ksenticnet[kor_word] = offset_sentic_dict[offset][:8] + ksemantics
                        similarity_dict[kor_word] = avg_simil

    for k, v in ksenticnet.items():
        idx = [i for i, x in enumerate(v[8:]) if x not in ksenticnet.keys()]
        for i in idx[::-1]:
            del ksenticnet[k][i+8]
    ksenticnet = {key: ksenticnet[key] for key in sorted(ksenticnet)}

    ksenticnet_file = open('ksenticnet_kaist.py', 'w', encoding='utf8')
    ksenticnet_file.write('ksenticnet = {}\n')
    for key, value in ksenticnet.items():
        ksenticnet_file.write('ksenticnet[\"' + key + '\"] = [' + sentic_to_string(value) + ']\n')

    ksenticnet_file.close()