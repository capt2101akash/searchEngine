from __future__ import print_function
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import collections
from collections import Counter
from nested_dict import nested_dict
import linecache
import os
import sys
import numpy as np
import csv
import json
import pickle
import re
import time
import math
from math import sqrt
import numpy as np
import Algorithmia

BASE_DIR = ''
DATASET_DIR = BASE_DIR + 'test/'
stop_words = set(stopwords.words('english'))
punc_marks = [',','.','/','(',')','{','}','@','#','-','_','?',':',';','`',"'",' ','``',"''","'",'bd', '!']
ps = PorterStemmer()
damp = 0.85
mapPr = 0.0
def synthesizeInput():
    temp = 0.0
    for i in range(0,2):
        query = raw_input("Enter your Search query.\n")
        tokens = []
        new_tokens = []
        stemmed_tokens = []
        tokens = word_tokenize(query)
        tokens = [items.lower() for items in tokens]
        for i, value in enumerate(tokens):
            if value not in stop_words and value not in punc_marks:
                new_tokens.append(value)
        new_tokens = [x for x in new_tokens if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]
        for w in new_tokens:
            stemmed_tokens.append(ps.stem(w))
        if stemmed_tokens:
            unpackIndex(stemmed_tokens)
        else:
            print('Please use a good query.')
    print(mapPr/2)
def unpackIndex(stemmed_tokens):
    with open('index.pkl', 'rb') as f:
        data = pickle.load(f)

    result = analyzeQuery(data, stemmed_tokens)
    if result:
        print(result)

def analyzeQuery(data, stemmed_tokens):
    unionFiles = []
    flag = 1
    for w in stemmed_tokens:
        for key, doc in data.items():
            if key==w:
                for key1, values in doc.items():
                    if key1 not in unionFiles:
                        unionFiles.append(key1)
    if not unionFiles:
        return 'No result Found'
    if 'IDF' in unionFiles:
        unionFiles.remove('IDF')
    docScore = {}
    dataList = []
    for docs in unionFiles:
        docScore[docs]={}
        for w in stemmed_tokens:
            if w in data:
                dataList = list(data[w])
                dataList.remove('IDF')

                if docs not in dataList:
                    docScore[docs].update({w:0})
                elif docs in dataList:
                    docScore[docs].update({w : data[w][docs][0]})
            else:
                return 'No Result'

    queryScore = {}
    freq = collections.Counter()
    freq.update(stemmed_tokens)
    for w in stemmed_tokens:
        if w in data:
            queryScore[w] = (freq.get(w)/len(stemmed_tokens)) * data[w]['IDF']
        else:
            return 'No Result'
    #print(docScore['akash'].get('en.13.31.311.2010.1.3'))
    cosScore = {}
    for docs, words in docScore.items():
        numerator = 0.0
        qrms = 0.0
        drms = 0.0
        denominator = 0.0
        for words1, qscore in queryScore.items():
            numerator += qscore * docScore[docs][words1]
            qrms += qscore*qscore
            drms += docScore[docs][words1]*docScore[docs][words1]
        denominator = math.sqrt(qrms * drms)
        cosScore[docs] = numerator/denominator
    maximum = max(cosScore, key=cosScore.get)
    count  = 0
    s = [(k, cosScore[k]) for k in sorted(cosScore, key=cosScore.get, reverse=True)]
    final_docs =[]
    for key, value in s:
        count+=1
        if count < 6:
            print('Your result document is :- ')
            print(key)
            final_docs.append(key)
            summarize(key, data)
        else:
            break
    relevant_docs = []
    f1 = csv.reader(open('convertcsv.csv'), delimiter = ';')
    for line in f1:
        if '126' in line[0] or '175' in line[0]:
            relevant_docs.append(line[2])
    total_size = len(final_docs)
    avgPr = 0.0
    count  = 0.0
    for i in range(0, total_size):
        if final_docs[i] in relevant_docs:
            count+=1
            avgPr+=count/(i+1)
    global mapPr
    print('Average Precision is :- \n')
    mapPr += avgPr/total_size
    print(mapPr)
def summarize(maximum, data):
    string = ''
    word_list=[]
    confList = []
    for path, subdirs, files in os.walk(DATASET_DIR):
        if maximum in files:
            with open(os.path.join(path, maximum)) as f:
                for num, line in enumerate(f, 1):
                    if num >= 4 and '</TEXT>' not in line and '<TEXT>' not in line:
                        string+=line
                    if re.match(r'</TEXT>', line, re.M|re.I):
                        break
    confList = languageDetect(string)
    word_list = nltk.sent_tokenize(string)
    main_dict = {}
    new_dict = {}
    string = []
    for i, value in enumerate(word_list):
        string = word_tokenize(value)
        if value != 'bd':
            new_list = []
            main_list = []
            for j, words in enumerate(string):
                main_list.append(words)
                words = words.lower()
                if words not in stop_words and words not in punc_marks :
                    new_list.append(words)
            new_list = [x for x in new_list if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]
            new_dict.update({i:new_list})
            main_dict.update({i:main_list})
    max_index = 0
    # for key, val in new_dict.items():
    #     if 'bd' in val:
    #         max_index = key
    # del new_dict[max_index]
    matrix = []
    rows = []
    columns = []
    numRows = 0
    for key, sentence in new_dict.items():
        numRows+=1
        rows.append(0.0)
    for key, sentence in new_dict.items():
        matrix.append(rows)
    matrix = np.array(matrix)
    for key, sentence in new_dict.items():
        num_sum1 = 0
        for key1, sentence1 in new_dict.items():
            freq = collections.Counter()
            freq.update(sentence)
            if matrix[key][key1] == 0 and matrix[key1][key] == 0 and key != key1:
                freq1 = collections.Counter()
                freq1.update(sentence1)
                root_x = 0
                root_y = 0
                for words in sentence:
                    if freq1.get(words):
                        num_sum1+=freq.get(words) * freq1.get(words) * pow(data[ps.stem(words)]['IDF'], 2)
                    else:
                        num_sum1 += 0
                    root_x += pow(freq.get(words) * data[ps.stem(words)]['IDF'], 2)

                root_x = sqrt(root_x)
                for words1 in sentence1:
                    root_y += pow(freq1.get(words1) * data[ps.stem(words1)]['IDF'], 2)
                root_y = sqrt(root_y)
                if root_y != 0 and root_x != 0:
                    res = num_sum1/(root_x * root_y)
                    matrix[key][key1] = res
                    matrix[key1][key] = res
                else:
                    matrix[key][key1] = 0
                    matrix[key1][key] = 0

    normRows = []
    rank = {}
    for i in range(0, numRows):
        rank.update({i:1/numRows})

    # for i in range(0, numRows):
    #     matrix[i]/=np.sum(matrix[i])
    #data[ps.stem(words)]['IDF']
    for rows in range(0,numRows):
        num_sum = 0.0
        den_sum = 0.0
        frac = 0.0
        for cols in range(0,numRows):
            if matrix[rows][cols] != 0.0:
                frac += (matrix[rows][cols]/np.sum(matrix[cols])) * rank.get(cols)
        rank.update({rows:frac})

    sent = max(rank, key = rank.get)
    summarySent = ' '.join(main_dict.get(sent))
    print('Summary is :-\n')
    print('"',summarySent,'"\n')
    print('Language used in file :- \n')

    if confList.result[0]['language'] == 'en':
        print('English')
    else:
        print(confList.result[0]['language'])

def languageDetect(sentence):
    inputSeq = {'sentence':sentence}
    client = Algorithmia.client('simb5w3XfR+KbICejsLj/niE7Md1')
    algo = client.algo('nlp/LanguageIdentification/1.0.0')
    return algo.pipe(inputSeq)
if __name__ == '__main__':
    synthesizeInput()
