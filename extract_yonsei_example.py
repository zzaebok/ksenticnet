import pickle
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from ksenticnet_kaist_duplicateHandling import ksenticnet
import time
import sys


def waitAndGetSoup(url, waitWhat = None):
    driver.get(url)
    if waitWhat != None:
        element = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, waitWhat)))
    html = driver.page_source
    soup = bs(html, 'html.parser')
    time.sleep(.2)
    return soup


driver = webdriver.Chrome('chromedriver.exe')

concept_list = []
example_dict ={}
modified_list = []
no_example_list = []

korean_wordnet_path = 'kwn_1.0/kwn_synset_list.tsv'
with open(korean_wordnet_path, 'r', encoding = 'utf8') as f:
    lines = f.read().split('\n')[1:]
    for line in lines:
        concept_list += line.split('\t')[3].replace(' ','').split(',')
    concept_list = list(set(concept_list))

for concept in concept_list:
    url = 'https://ilis.yonsei.ac.kr/ysdic/word/YT/'+ concept + '/0'
    soup = waitAndGetSoup(url)
    word = soup.find('div', {'class':'page-header'}).text.replace(' ','').replace('\n', '')
    if word == '':
        #case 1: multiple meanings
        url = 'https://ilis.yonsei.ac.kr/ysdic/word/YT/'+ concept + '/1'
        soup = waitAndGetSoup(url)
        word = soup.find('div', {'class': 'page-header'}).text.replace(' ', '').replace('\n', '')
        if word == '':
            #case 2: verb
            url = 'https://ilis.yonsei.ac.kr/ysdic/word/YT/'+ concept + '다/0'
            soup = waitAndGetSoup(url)
            word = soup.find('div', {'class': 'page-header'}).text.replace(' ', '').replace('\n', '')
            if word =='':
                #case 3: verb, multiple meanings
                url = 'https://ilis.yonsei.ac.kr/ysdic/word/YT/' + concept + '다/1'
                soup = waitAndGetSoup(url)
                word = soup.find('div', {'class': 'page-header'}).text.replace(' ', '').replace('\n', '')
                if word =='':
                    #case 4: doesn't exist on dictionary -> naver dictionary
                    url = 'https://ko.dict.naver.com/#/search?query=' + concept + '&range=example'
                    soup = waitAndGetSoup(url, 'footer')
                    try:
                        first_example = driver.find_element_by_xpath('//*[@id="searchPage_example"]/ul/li[1]/div[1]/p')
                        theWord = first_example.find_element_by_tag_name('strong').text
                        first_example = first_example.text
                        index = first_example.find(theWord)
                        left = first_example[:index]
                        right = first_example[index + len(theWord):]
                        example_dict[concept] = (left, right)
                    except Exception as e:
                        url = 'https://ko.dict.naver.com/#/search?query=' + concept + '다&range=example'
                        soup = waitAndGetSoup(url, 'footer')
                        try:
                            first_example = driver.find_element_by_xpath(
                                '//*[@id="searchPage_example"]/ul/li[1]/div[1]/p')
                            theWord = first_example.find_element_by_tag_name('strong').text
                            first_example = first_example.text
                            index = first_example.find(theWord)
                            left = first_example[:index]
                            right = first_example[index + len(theWord):]
                            example_dict[concept] = (left, right)
                            modified_list.append(word)
                        except:
                            no_example_list.append(concept)
                    continue
                else:
                    modified_list.append(concept)
            else:
                modified_list.append(concept)
    added_list = soup.find_all('li')
    example_flag = False
    for adds in added_list:
        if adds.find('strong').text == '예문: ':
            example_flag = True
            sentence = adds.text
            theWord = adds.find('u').text
            slash = sentence.find('/')
            if slash!= -1:
                sentence = sentence[:slash]
            sentence = sentence.replace('예문: ','')
            index = sentence.find(theWord)
            left = sentence[:index]
            right = sentence[index+len(theWord):]
            example_dict[concept] = (left, right)
            break

    if example_flag == False:
        no_example_list.append(concept)

### there was example sentences but didn't crawled for some reason
example_dict['축제'] = ('','가 열리다')
example_dict['이'] = ('',', 피해라.')
example_dict['어구'] = ('겨울의 ','')
example_dict['진피'] = ('목 피부는 ', '층이 매우 얇고 목의 근육이 피부 바로 아래에 존재하여 주름이 쉽게 발생할 수 있다.')
example_dict['의론'] = ('이러니저러니 ', '이 분분하다.')
example_dict['표식'] = ('고속도 도로의 도로', '.')
example_dict['심통'] = ('', '이 나다')
example_dict['프랑'] = ('그 그림은 내가 프랑스에서 유학 중일 때 200', '에 샀던 것입니다.')
example_dict['속임질하'] = ('아버지는 친구에게 ','을 당해 많은 돈을 날렸다.')
example_dict['부주'] = ('','전 상서.')
example_dict['골키퍼'] = ('강하게 슛을 날렸지만 ','의 손에 걸리고 말았다.')
example_dict['원앙새'] = ('옛날부터 내외간 금실 좋은 것을 ','에 비하지 않았어요.')
example_dict['담장'] = ('','을 넘다')
example_dict['간호원'] = ('젊은 의사가 혈압기를 든 ','과 함께 다가와서 물었다.')
example_dict['미음'] = ('','을 쑤다 ')
example_dict['반편'] = ('선형은 한 걸음쯤 그 모친의 뒤에 피하여 한편 귀와 몸의 ','이 그 모친에게 가리었다.')
example_dict['무우'] = ('','시래기와 배추 시래기.')
example_dict['텔레비'] = ('래일 아침에 저금소에서 ','값을 찾아다 주십시오.')
example_dict['귤나무'] = ('이제 ','가 성목이 되었으니 귤 생산량이 많이 늘어날 것이다.')

### no example sentences, Used definition sentence.
example_dict['온스'] = ('상용 ','는 1파운드의 16분의 1로 28.35그램에 해당하고, 금ㆍ은ㆍ약제용으로는 1파운드의 12분의 1로 약 31.1035그램에 해당한다.')
example_dict['흥분제'] = ('','중추 신경을 자극하여 신경 계통 및 뇌와 심장의 기능을 활발하게 하는 약. 장뇌(樟腦), 에틸 에테르, 카페인, 포도주 따위이다.')
example_dict['심수'] = ('', '마음으로 받아들여 깨달음.')
example_dict['탄착'] = ('','탄알이 명중함. 또는 그런 지점.')
example_dict['강배'] = ('','강에서 쓰는 배. 배의 밑이 평평하게 되어 있다.')
example_dict['인터페론'] = ('','바이러스에 감염된 동물 세포가 생성하는 당단백질. 바이러스의 감염과 증식을 저지하는 작용을 한다. 유전 공학의 발달로 대량 생산되며, 비형(B型) 간염이나 헤르페스 따위의 바이러스 질병의 치료에 쓰인다.')
example_dict['세일러'] = ('','인도에서 회교도들이 어깨에 두르는 면 스카프.')
example_dict['밀리리터'] = ('','미터법에 의한 부피의 단위. 1밀리리터는 1리터의 1,000분의 1이다. 기호는 mL, ml')
example_dict['날개돋치'] = ('','상품이 시세를 만나 빠른 속도로 팔려 나가다')
example_dict['라이거'] = ('','수사자와 암호랑이 사이에 태어난 잡종. 몸은 사자보다 약간 크며, 색깔은 사자와 비슷하나 좀 어둡고, 갈색의 줄무늬가 있는데 또렷하지는 않다. 생식 능력이 없다.')
example_dict['바트'] = ('질레 가멍 춤 ','민 안 뒈지.(제주)')
example_dict['생혈'] = ('','살아 있는 동물의 몸에서 갓 빼낸 피')


with open('example_dict.pkl', 'wb') as f:
    pickle.dump(example_dict, f)

with open('modified_list.pkl', 'wb') as f:
    pickle.dump(modified_list, f)

with open('no_example_list.pkl', 'wb') as f:
    pickle.dump(no_example_list, f)

