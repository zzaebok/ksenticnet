from senticnet5 import senticnet
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
import pickle
import numpy as np
import math

############ basic functions #############
def average_sentic(value1, value2): #both are lists
    frequency = bring_frequency()
    value1[1] = float(value1[1])
    value1[2] = float(value1[2])
    value1[3] = float(value1[3])
    value1[4] = float(value1[4])
    value1[8] = float(value1[8])
    value2[1] = float(value2[1])
    value2[2] = float(value2[2])
    value2[3] = float(value2[3])
    value2[4] = float(value2[4])
    value2[8] = float(value2[8])

    if value1[0]=='DUPLICATE' and value2[0] == 'DUPLICATE':
        value1[9] = float(value1[9])
        value2[9] = float(value2[9])
        pl = value1[1] * value1[9] + value2[1] * value2[9]
        at = value1[2] * value1[9] + value2[2] * value2[9]
        se = value1[3] * value1[9] + value2[3] * value2[9]
        ap = value1[4] * value1[9] + value2[4] * value2[9]
        pv = value1[8] * value1[9] + value2[8] * value2[9]
        freq = (value1[9]+value2[9])
    elif value1[0]!='DUPLICATE' and value2[0] == 'DUPLICATE':
        value2[9] = float(value2[9])
        pl = value1[1] * frequency[value1[0]] + value2[1] * value2[9]
        at = value1[2] * frequency[value1[0]] + value2[2] * value2[9]
        se = value1[3] * frequency[value1[0]] + value2[3] * value2[9]
        ap = value1[4] * frequency[value1[0]] + value2[4] * value2[9]
        pv = value1[8] * frequency[value1[0]] + value2[8] * value2[9]
        freq = (frequency[value1[0]]+value2[9])
    elif value1[0]=='DUPLICATE' and value2[0] != 'DUPLICATE':
        value1[9] = float(value1[9])
        pl = value2[1] * frequency[value2[0]] + value1[1] * value1[9]
        at = value2[2] * frequency[value2[0]] + value1[2] * value1[9]
        se = value2[3] * frequency[value2[0]] + value1[3] * value1[9]
        ap = value2[4] * frequency[value2[0]] + value1[4] * value1[9]
        pv = value2[8] * frequency[value2[0]] + value1[8] * value1[9]
        freq = (frequency[value2[0]] + value1[9])
    else:
        pl = value1[1] * frequency[value1[0]] + value2[1] * frequency[value2[0]]
        at = value1[2] * frequency[value1[0]] + value2[2] * frequency[value2[0]]
        se = value1[3] * frequency[value1[0]] + value2[3] * frequency[value2[0]]
        ap = value1[4] * frequency[value1[0]] + value2[4] * frequency[value2[0]]
        pv = value1[8] * frequency[value1[0]] + value2[8] * frequency[value2[0]]
        freq = (frequency[value1[0]]+frequency[value2[0]])

    sentic = [pl / freq, at / freq, se / freq, ap / freq]
    return ['DUPLICATE'] + [round(pl / freq, 3), round(at / freq,3), round(se / freq,3), round(ap / freq,3)] + sentic_value_to_sentiments(sentic) + [value1[7], round(pv/freq,3)]




def sentic_value_to_sentiments(sentic_value):
    sentiment_dict = {
    0: ['#grief','#sadness','#pensiveness','#serenity','#joy','#ecstasy'],
    1: ['#amazement','#surprise', '#distraction','#interest','#anticipation','#vigilance'],
    2: ['#terror','#fear','#apprehension','#annoyance','#anger','#rage'],
    3: ['#loathing','#disgust','#boredom','#acceptance','#trust','#admiration']
    }

    abs_value = [abs(number) for number in sentic_value]
    first = np.argmax(abs_value)
    abs_value[first] = 0
    second = np.argmax(abs_value)
    if sentic_value[second] == 0:
        second = first
    first_index = math.floor(sentic_value[first]*3)+3
    second_index = math.floor(sentic_value[second]*3)+3
    return [sentiment_dict[first][first_index], sentiment_dict[second][second_index]]


def bring_frequency():
    affectnet_frequency_dict = {}
    with open('affectnet_dict.pkl', 'rb') as p:
        affectnet_frequency_dict = pickle.load(p)
    return affectnet_frequency_dict

def value_to_string(value):
    #input : dictionary value
    #output : string value by concatenating dictionary value elements
    string = ''
    for s in value:
        string += '\''+str(s)+'\', '
    return string[:-2]

def synset_cluster(except_word, synset):
    #input: the keyword and its synset
    #output: hierarchical list of the synsets

    # [
    # [[level0][level1][level2][level3]],
    # [[level0][level1][level2][level3]],
    # [[level0][level1][level2][level3]],
    # [[level0][level1][level2][level3]]
    # ]

    result = [[], [], [], []]
    level0 = [lemma.name().lower() for lemma in synset.lemmas()]
    level0.remove(except_word)
    h1 = []
    h2 = []
    h3 = []
    level1 = []
    level2 = []
    level3 = []
    h1 = synset.hypernyms()
    for x in h1:
        h2 += x.hypernyms()
        level1 += [lemma.name().lower() for lemma in x.lemmas()]
    for x in h2:
        h3 += x.hypernyms()
        level2 += [lemma.name().lower() for lemma in x.lemmas()]
    for x in h3:
        level3 += [lemma.name().lower() for lemma in x.lemmas()]
    result[0] = level0
    result[1] = level1
    result[2] = level2
    result[3] = level3
    return result

def making_korean_wordnet_key_list(korean_wordnet_directory):
    korean_wordnet_key_list = []
    f = open(korean_wordnet_directory, 'r', encoding='utf8')
    line_list = f.read().split('\n')[1:]
    for line in line_list:
        korean_wordnet_key_list.append(line.split('\t')[0])
    f.close()
    return korean_wordnet_key_list

############## direct mapping ##############
def making_word_synset_dict(korean_wordnet_directory):
    #input: korean_wordnet_directory - 00018158-v	Verb	uprise, rise, arise, turn_out, get_up	기상하
    #output: word_synset_dict[uprise] = [00018158-v, ...]
    word_synset_dict = {}
    f = open(korean_wordnet_directory, 'r', encoding='utf8')
    line_list = f.read().split('\n')[1:]
    for line in line_list:
        word_list = line.split('\t')[2].replace(' ','').split(',')
        for word in word_list:
            synset_offset = line.split('\t')[0]
            if word not in word_synset_dict:
                word_synset_dict[word] = [synset_offset]
            else:
                word_synset_dict[word].append(synset_offset)
    f.close()
    return word_synset_dict

def treat_duplicates(offset_sentic_dict, duplicate_dict, matched_list):
    frequency = bring_frequency()
    for duplicate_synset in list(duplicate_dict.keys()):
        duplicate_value = duplicate_dict[duplicate_synset]
        tmp_matched_list = []
        tmp_matched_list.append(offset_sentic_dict[duplicate_synset][0])
        flag = False
        synset_ID = duplicate_synset
        freq = frequency[offset_sentic_dict[synset_ID][0]]  # find by concept
        polarity = offset_sentic_dict[synset_ID][7]
        polarity_value = float(offset_sentic_dict[synset_ID][8]) * freq
        pl_value = float(offset_sentic_dict[synset_ID][1]) * freq
        at_value = float(offset_sentic_dict[synset_ID][2]) * freq
        se_value = float(offset_sentic_dict[synset_ID][3]) * freq
        ap_value = float(offset_sentic_dict[synset_ID][4]) * freq
        for duplicate in duplicate_value:
            tmp_freq = frequency[duplicate[0]]
            freq += tmp_freq
            if duplicate[7] != polarity:  # polarity positive vs negative case -> remove
                del offset_sentic_dict[synset_ID]
                del duplicate_dict[synset_ID]
                flag = True
                break
            else:  # polarity safe
                tmp_matched_list.append(duplicate[0])
                polarity_value += float(duplicate[8]) * tmp_freq
                pl_value += float(duplicate[1]) * tmp_freq
                at_value += float(duplicate[2]) * tmp_freq
                se_value += float(duplicate[3]) * tmp_freq
                ap_value += float(duplicate[4]) * tmp_freq
        if flag:
            continue
        else:
            matched_list += tmp_matched_list
            sentic_value = [round(pl_value / freq, 3), round(at_value / freq,3), round(se_value / freq, 3), round(ap_value / freq, 3)]
            two_sentiments = sentic_value_to_sentiments(sentic_value)
            offset_sentic_dict[synset_ID] = ['DUPLICATE'] + sentic_value + two_sentiments + [polarity, round(polarity_value / freq, 3)] + [freq]
    return

def making_offset_sentic_dict(word_synset_dict, matched_list):
    #input: word_synset_dict - word_synset_dict[uprise] = [00018158-v, ...]
    #output: offset_sentic_dict["00044455-n"] = ['emergence', '#joy', '#surprise', 'positive', '0.726', 'appearance', 'start', 'casus_belli', 'beginning', 'egress']

    duplicate_dict = {}
    offset_sentic_dict = {}
    for word in word_synset_dict.keys():
        flag = False
        if word in senticnet.keys():
            synsets = word_synset_dict[word]
            try:
                candidates = [ synset_cluster(word, wn.synset_from_pos_and_offset(synset_id[-1], int(synset_id[:8].lstrip('0')))) for synset_id in synsets ]
                # candidates = [[~~],[~~],[~~],[~~],[~~],[~~]]
            except:
                continue
            for x in range(4):
                for y in range(len(candidates)):
                    for candidate in candidates[y][x]:
                        if candidate in senticnet[word]:
                            if synsets[y] not in offset_sentic_dict.keys():
                                offset_sentic_dict[synsets[y]] = [word]+senticnet[word]
                                matched_list.append(word)
                            else:
                                ### may have duplicates ###
                                if synsets[y] not in duplicate_dict.keys():
                                    duplicate_dict[synsets[y]] = [[word]+senticnet[word]]
                                else:
                                    duplicate_dict[synsets[y]].append([word]+senticnet[word])
                            flag = True
                            break

                if flag: break
                # level stopped
    treat_duplicates(offset_sentic_dict, duplicate_dict, matched_list)
    return offset_sentic_dict

############## lesk algorithm ##############
def apply_lesk_algorithm(offset_sentic_dict, matched_list):
    #making another offset_sentic_dict for rest of concepts and concat it to the existed one.
    #input: offset_sentic_dict["00044455-n"] = ['emergence', '#joy', '#surprise', 'positive', '0.726', 'appearance', 'start', 'casus_belli', 'beginning', 'egress'] // semantics might not included
    #output: last_offset_sentic_dict["00044455-n"] = ['emergence', '#joy', '#surprise', 'positive', '0.726', 'appearance', 'start', 'casus_belli', 'beginning', 'egress']
    duplicate_dict = {}
    korean_wordnet_key_list = making_korean_wordnet_key_list(korean_wordnet_directory)
    tmp_offset_sentic_dict = {}
    for matched in list(set(matched_list)):
        # already direct mapped keys are deleted.
        del senticnet[matched]

    for key, value in senticnet.items():
        flag = False
        context = key
        for x in range(8, 13):
            context += senticnet[key][x] + ' '

        try:
            ss = lesk(context, key)
            offset = str(ss.offset()).zfill(8) + '-' + ss.pos()
            flag = True
        except AttributeError:
            for v in value[8:13]:   #sequential search because semantics are arranged based on cosine similarity
                try:
                    ss = lesk(context, v)
                    offset = str(ss.offset()).zfill(8) + '-' + ss.pos()
                    flag = True
                except AttributeError:
                    continue
        if not flag: continue
        #no match in key and all its semantics

        if offset not in tmp_offset_sentic_dict.keys(): #not yet included
            if offset in korean_wordnet_key_list:   #those we can handle
                tmp_offset_sentic_dict[offset] = [key] + value
        else:
            if offset not in duplicate_dict.keys():
                duplicate_dict[offset] = [[key]+value]
            else:
                duplicate_dict[offset].append([key]+value)
            #duplicates mean that it is already directly mapped, so we don't have to worry about them
    treat_duplicates(tmp_offset_sentic_dict, duplicate_dict, matched_list)
    for synset in list(tmp_offset_sentic_dict.keys()):
        value = tmp_offset_sentic_dict[synset]
        if synset not in offset_sentic_dict:
            offset_sentic_dict[synset] = value
        else:
            if (value[7]=='positive' and offset_sentic_dict[synset][7] =='negative') or (value[7]=='negative' and offset_sentic_dict[synset][7] =='positive'):
                del tmp_offset_sentic_dict[synset]
                del offset_sentic_dict[synset]
                continue
            else:
                offset_sentic_dict[synset] = average_sentic(value, offset_sentic_dict[synset])

    #offset weighted average
    return offset_sentic_dict


############## making ksenticnet ###########
if __name__ == "__main__":

    # constant variables
    korean_wordnet_directory = 'pwn_synset_list.tsv'
    matched_list = [] #to control duplicates
    ksenticnet = {}
    ksenticnet_file = open('ksenticnet_pusan_duplicateHandling.py', 'w', encoding='utf8')
    ksenticnet_file.write('ksenticnet = {}\n')

    word_synset_dict = making_word_synset_dict(korean_wordnet_directory)
    offset_sentic_dict = making_offset_sentic_dict(word_synset_dict, matched_list)
    last_offset_sentic_dict = apply_lesk_algorithm(offset_sentic_dict, matched_list)

    korean_wordnet_file = open(korean_wordnet_directory, 'r', encoding='utf8')
    line_list = korean_wordnet_file.read().split('\n')[1:]
    for line in line_list:
        offset = line.split('\t')[0]
        if offset in last_offset_sentic_dict.keys():
            korean_synset = line.split('\t')[3].replace(' ','').split(',')
            for kword in korean_synset:
                if kword not in ksenticnet.keys():
                    ksemantics = korean_synset[:] #list copy
                    if len(korean_synset)!=1:
                        ksemantics.remove(kword)
                    else:
                        ksemantics = []
                    ksenticnet[kword] = last_offset_sentic_dict[offset][1:9] + ksemantics
                else:
                    print(kword, ' is already in ksenticnet')
    for key,value in ksenticnet.items():
        ksenticnet_file.write('ksenticnet[\"'+key+'\"] = ['+value_to_string(value)+']\n')
    ksenticnet_file.close()

# offset duplicated : 하나의 offset에 여러 가지 sentic value가 붙는 현상 //근데 하나의 단어가 아니라서 weighted average도 못씀.
# key duplicated in ksenticnet : 하나의 한국어 단어에 여러 가지 sentic value가 붙는 현상 ( 하나의 한국어 단어에 여러 가지 offset이 붙어서 발생함 )