import pickle
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from ksenticnet_kaist_duplicateHandling import ksenticnet
import time
import sys

global driver

def waitAndGetSoup(url, waitWhat = None):
    driver.get(url)
    if waitWhat != None:
        element = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, waitWhat)))
    html = driver.page_source
    soup = bs(html, 'html.parser')
    time.sleep(.2)
    return soup


driver = webdriver.Chrome('chromedriver.exe')

concept_list = list(ksenticnet.keys())
example_dict ={}
modified_list = []
no_example_list = []

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

example_dict['축제'] = ('','가 열리다')
example_dict['이'] = ('',', 피해라.')
example_dict['어구'] = ('겨울의 ','')
example_dict['진피'] = ('목 피부는 ', '층이 매우 얇고 목의 근육이 피부 바로 아래에 존재하여 주름이 쉽게 발생할 수 있다.')
example_dict['의론'] = ('이러니저러니 ', '이 분분하다.')
example_dict['표식'] = ('고속도 도로의 도로', '.')
example_dict['심통'] = ('', '이 나다')
example_dict['프랑'] = ('그 그림은 내가 프랑스에서 유학 중일 때 200', '에 샀던 것입니다.')
example_dict['속임질하'] = ('아버지는 친구에게 ','을 당해 많은 돈을 날렸다.')

### no example sentences, Used definition sentence.
example_dict['온스'] = ('상용 ','는 1파운드의 16분의 1로 28.35그램에 해당하고, 금ㆍ은ㆍ약제용으로는 1파운드의 12분의 1로 약 31.1035그램에 해당한다.')
example_dict['흥분제'] = ('','중추 신경을 자극하여 신경 계통 및 뇌와 심장의 기능을 활발하게 하는 약. 장뇌(樟腦), 에틸 에테르, 카페인, 포도주 따위이다.')
example_dict['심수'] = ('', '마음으로 받아들여 깨달음.')


with open('example_dict.pkl', 'wb') as f:
    pickle.dump(example_dict, f)

with open('modified_list.pkl', 'wb') as f:
    pickle.dump(modified_list, f)

with open('no_example_list.pkl', 'wb') as f:
    pickle.dump(no_example_list, f)

