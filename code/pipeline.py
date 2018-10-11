import nltk
import re
import numpy
from nltk.corpus import PlaintextCorpusReader
import os
import csv
import copy
import random
import argparse

from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
nltk.download("stopwords")

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--stopword", help="remove stopwords before testing", action="store_true")
args = parser.parse_args()

# Get tweets
tweets = []
with open('../data/tweets.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        for i in range(len(row)):
            if (len(row[i]))>1:
                tweets.append(row[i])

############################################

# helper preprocessing functions

tokenizer = TreebankWordTokenizer()

def tokenize(s):
    return tokenizer.tokenize(s)

def decap_and_depunct(words):
    no_caps_no_punct = []
    no_caps_no_punct = [w.lower() for w in words if w.isalpha() or w == '!']
    return no_caps_no_punct

def remove_stopwords(words):
    no_stw = []
    no_stw = [w for w in words if w.lower() not in stopwords.words('english')]
    return no_stw

def remove_url(words):
    for i in range(len(words)):
        if words[i]=='https' or words[i] == 'http' or words[i] == 'www':
            return words[0:i-1]
    return words

#############################################

# tokenize & preprocess for naive bayes frequency analysis

tokenized = []

for tweet in tweets: 
        tokens = tokenize(tweet)
        tokens = decap_and_depunct(tokens)
        tokens = remove_url(tokens)
        if args.stopword:
            tokens = remove_stopwords(tokens)
        tokenized.append(tokens)

############################################

# Create randomization
random_index = range(4000)
random.shuffle(random_index)

# Create training and test datasets for 10-fold cross validation
tr_data_fold = []
X_fold = []
y_fold = []
test_data_fold = []
test_labels_fold = []
with open('../data/preprocessed_tweets_labels.csv', 'rb') as labelFile:
    reader = csv.reader(labelFile)
    for fold in range(10):
        labelFile.seek(0)
        tr_data = []
        X = []
        y = []
        test_data = []
        test_labels = []
        for i in range(4000):
            currentLabel = next(reader)
            feature = tokenized[i]
            label = 0
            if int(currentLabel[0]) == 4:
                label = -1
            elif int(currentLabel[0]) == 1:
                label = 1
            else:
                continue
            start_index = fold*400
            if i in random_index[start_index:start_index + 400]:
                test_data.append((feature, label))
                test_labels.append(label)
            else:
                tr_data.append((feature, label))
                X.append(feature)
                y.append(label)
        tr_data_fold.append(tr_data)
        X_fold.append(X)
        y_fold.append(y)
        test_data_fold.append(test_data)
        test_labels_fold.append(test_labels)
            
#############################################
