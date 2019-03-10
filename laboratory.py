#from ksenticnet_pusan_duplicateHandling import ksenticnet as ks1
from ksenticnet_kaist import ksenticnet
from konlpy.tag import Kkma
import random
import traceback

#ksenticnet = ks1

'''
keys = list(ksenticnet.keys())
random.shuffle(keys)
ksenticnet =  [(key, ksenticnet[key]) for key in keys]
# if abs(float(ksenticnet[key][7]))>=0.25
size = 100
half = size/2
i = 0
j = 0
pos_list = []
neg_list = []
for key, value in ksenticnet:
    if i==half and j==half:
        break
    if float(value[7])>0:
        if i == half:
            continue
        pos_list.append(key)
        i += 1
    else:
        if j == half:
            continue
        neg_list.append(key)
        j += 1

candidates = []
candidates = pos_list + neg_list
random.shuffle(candidates)
counter = 0
for word in candidates:
    print(word)
    polarity = input('극성?: ')
    if polarity == '0' and word in neg_list:
        counter += 1
    elif polarity == '1' and word in pos_list:
        counter += 1

print('정답율: ', float(counter/size))
'''
'''
ksenticnet = dict()
f = open('knu.txt', 'r', encoding='utf8')
line_list = f.read().split('\n')
for line in line_list:
    ksenticnet[line.split('\t')[0]] = [0,0,0,0,0,0,0,line.split('\t')[1]]
f.close()
'''




tokenizer = Kkma()
filename = 'ratings_test.txt'
f = open(filename, 'r', encoding='utf8')
line_list = f.read().split('\n')[1:]

false_pos = 0
false_neg = 0
true_pos = 0
true_neg = 0

pos = 0
neg = 0

for line in line_list:
    score = 0
    flag = False
    if filename.startswith('r'):
        review = line.split('\t')[1]
        polarity = int(line.split('\t')[2])
    else:
        review = line.split('\t')[0]
        polarity = int(line.split('\t')[1])
    candidates = []
    for token in [tuple[0] for tuple in tokenizer.pos(review)]: #여기도 같이 보자.
        if token in ksenticnet.keys():
            flag = True #able to evaluate
            candidates.append((token, float(ksenticnet[token][7])))
            #score += float(ksenticnet[token][7])
            if float(ksenticnet[token][7]) < 0:
                score -=1
            else:
                score +=1
    if flag:
        if pos == 1000 and neg == 1000:
            break
        if (pos == 1000 and polarity == 1) or (neg == 1000 and polarity == 0):
            continue

        if polarity == 0 and score <0:
            true_neg += 1
            neg += 1
        elif polarity == 1 and score > 0:
            true_pos += 1
            pos +=1
        elif polarity == 1 and score <0:
            false_neg += 1
            pos +=1
        elif polarity ==0 and score >0:
            false_pos += 1
            neg+=1
            '''
        else:
            print(review,' 분석: ', candidates , ' 정답: ', polarity, ' 예측: ', score)
            '''

total_counter = true_neg+true_pos+false_neg+false_pos
print('분석 가능한 리뷰 개수: ', total_counter)
print('정답 개수: ', true_pos+true_neg)
print('전체 정답율: ', float((true_pos+true_neg)/total_counter)*100, '%')
print()
print('총 긍정 개수: ', true_pos+false_neg, ' 총 부정 개수: ',true_neg+false_pos)
print('맞은 긍정 개수: ', true_pos, ' 맞은 부정 개수: ',true_neg)
print('긍정 정답율: ',float(true_pos/(true_pos+false_neg)),' 부정 정답율: ',float(true_neg/(true_neg+false_pos)))
p = true_pos/(true_pos+false_pos)
r = true_pos/(true_pos+false_neg)
f1 = (2*p*r)/(p+r)
print('P: ',p, ' R: ', r ,' F1: ', f1)
f.close()


#남은 과제. : 중복 처리 기술 / more crawling / 아무래도 kaist껏만 하는게...