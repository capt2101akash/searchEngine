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

BASE_DIR = ''
DATASET_DIR = BASE_DIR + 'test/'
stop_words = set(stopwords.words('english'))
punc_marks = [',','.','/','(',')','{','}','@','#','-','_','?',':',';','``',"''",' ']
ps = PorterStemmer()
'''
clean_values contains the final index in which the format is {word:{file:[tf*idf, wordFreq, totalWords, pos...]}

'''
def tf(clean_values, count):
    for key, values in clean_values.items():
        totalWords = 0
        for key1, values1 in values.items():
            termFreq = values1[0]/values1[1]
            values1.insert(0, termFreq)
            values.update({key1:values1})
        clean_values.update({key:values})
    idf(clean_values, count)

def idf(value, count):
    list1 = []
    clean_values = {}

    for key, values in value.items():
        inverseDocFreq = 1 + math.log10(count / len([items for items in values if items]))
        tfidf = 0.0
        for key1, values1 in values.items():
            values1.insert(1, values1[0] * inverseDocFreq)
            values.update({key1:values1})
        clean_values.update({key:values})
        clean_values[key]['IDF'] = inverseDocFreq
    print(clean_values)
    output = open('index.pkl','wb')
    pickle.dump(clean_values, output)
    output.close()

def list_dir(DATASET_DIR):
    start_line = 0
    end_line = 0
    values = {}
    count = 0
    word_list = []
    for path, subdirs, files in os.walk(DATASET_DIR):
        for name in files:
            with open(os.path.join(path, name)) as f:
                string = ''
                count +=1
                for num, line in enumerate(f, 1):
                    if num >= 4 and '</TEXT>' not in line and '<TEXT>' not in line:
                        string+=line
                    if re.match(r'</TEXT>', line, re.M|re.I):
                        break
            word_list = [string, len(string.split())]
            values.update({os.path.basename(os.path.normpath(os.path.join(path, name))) : word_list})
    return values

final_dict={}

def do_tokenize():
    values = {}
    values = list_dir(DATASET_DIR)
    clean_values = {}
    count = 0
    for key, val in values.items():
        count +=1
        tokens = word_tokenize(val[0])
        tokens = [item.lower() for item in tokens]
        new_tokens = []
        pos = {}
        dup_tokens = []

        for i, value in enumerate(tokens):
            if value not in stop_words and value not in punc_marks:
                new_tokens.append(value)
        new_tokens = [x for x in new_tokens if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]

        for index, value1 in enumerate(new_tokens):
            if value1 not in pos:
                pos.setdefault(value1, [])
            pos[value1].append(index)
        freq = collections.Counter()
        freq.update(new_tokens)

        for w, list1 in pos.items():
            stem_words = ps.stem(w)
            word_freq = freq.get(w)
            list1.insert(0, word_freq)
            list1.insert(1, val[1])
            if stem_words not in clean_values:
                clean_values[stem_words] = {key:list1}
            if key not in clean_values[stem_words]:
                clean_values[stem_words].update({key:list1})
        if count%200 == 0:
            print(count)
    tf(clean_values, count)

do_tokenize()
