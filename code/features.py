import os
import csv
import numpy
from sklearn import svm
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# List of English stopwords
stopWords = list(stopwords.words("english"))

# labeledTweets are of the form [label, length, [tweet words]]
labeledTweets = []
labeledTweetsNoStopWords = []

# Dictionaries matching unigrams/bigrams/trigrams with frequencies
happyUnigrams = {}
happyBigrams = {}
happyTrigrams = {}
unhappyUnigrams = {}
unhappyBigrams = {}
unhappyTrigrams = {}

happyUnigramsNoStopWords = {}
happyBigramsNoStopWords = {}
happyTrigramsNoStopWords = {}
unhappyUnigramsNoStopWords = {}
unhappyBigramsNoStopWords = {}
unhappyTrigramsNoStopWords = {}

# Dictionary of number of each type of label
labelNumbers = {'happy':0, 'unhappy':0, 'neutral':0, 'total':0}

# Oraanize data into labeledTweets
with open('../data/preprocessed_tweets_modified.csv', 'r') as tweetFile:
    with open('../data/preprocessed_tweets_labels.csv', 'r') as labelFile:
        tweetReader = csv.reader(tweetFile)
        labelReader = csv.reader(labelFile)
        currentTweet = next(tweetReader)
        for label in labelReader:
            length = 0
            lengthNoStopWords = 0
            modTweet = list(currentTweet)
            modTweetNoStopWords = list(currentTweet)
            for i in range(0, len(currentTweet)):
                if(currentTweet[i] != ''):
                    modTweet[i] = currentTweet[i][2:-1]
                    length += 1 
                    if(currentTweet[i][2:-1] not in stopWords):
                        modTweetNoStopWords[lengthNoStopWords] = currentTweet[i][2:-1]
                        lengthNoStopWords += 1
                else:
                    break
            modLabel = label[0]
            if (label[0] == ''):
                modLabel = '0'
            labeledTweets.append([int(modLabel), length, modTweet[:length]])
            labeledTweetsNoStopWords.append([int(modLabel), lengthNoStopWords, modTweetNoStopWords[:lengthNoStopWords]])
            currentTweet = next(tweetReader)
            # Find label numbers
            if (int(modLabel) == 0):
                labelNumbers['neutral'] += 1
            elif (int(modLabel) == 4):
                labelNumbers['unhappy'] += 1
            else:
                labelNumbers['happy'] += 1
            labelNumbers['total'] += 1

# Find common unigrams bigrams, and trigrams (including stopwords)
for tweet in labeledTweets:
    for i in range(len(tweet[2])):
        # Find common unigrams
        unigram = tweet[2][i]
        if (tweet[0] == 1 or tweet[0] == 2):
            if (unigram in happyUnigrams):
                happyUnigrams[unigram] += 1
            else:
                happyUnigrams[unigram] = 1
        if (tweet[0] == 4):
            if (unigram in unhappyUnigrams):
                unhappyUnigrams[unigram] += 1
            else:
                unhappyUnigrams[unigram] = 1  
        # Find common bigrams
        if (i > 0):
            bigram = tweet[2][i-1] + ' ' + tweet[2][i]
            if (tweet[0] == 1 or tweet[0] == 2):
                if (bigram in happyBigrams):
                    happyBigrams[bigram] += 1
                else:
                    happyBigrams[bigram] = 1
            if (tweet[0] == 4):
                if (bigram in unhappyBigrams):
                    unhappyBigrams[bigram] += 1
                else:
                    unhappyBigrams[bigram] = 1  
        # Find common trigrams
        if (i > 2):
            trigram = tweet[2][i-2] + ' ' + tweet[2][i-1] + ' ' + tweet[2][i]
            if (tweet[0] == 1 or tweet[0] == 2):
                if (trigram in happyTrigrams):
                    happyTrigrams[trigram] += 1
                else:
                    happyTrigrams[trigram] = 1
            if (tweet[0] == 4):
                if (trigram in unhappyTrigrams):
                    unhappyTrigrams[trigram] += 1
                else:
                    unhappyTrigrams[trigram] = 1  

# Find common unigrams bigrams, and trigrams (excluding stopwords)
for tweet in labeledTweetsNoStopWords:
    for i in range(len(tweet[2])):
        # Find common unigrams
        unigram = tweet[2][i]
        if (tweet[0] == 1 or tweet[0] == 2):
            if (unigram in happyUnigramsNoStopWords):
                happyUnigramsNoStopWords[unigram] += 1
            else:
                happyUnigramsNoStopWords[unigram] = 1
        if (tweet[0] == 4):
            if (unigram in unhappyUnigramsNoStopWords):
                unhappyUnigramsNoStopWords[unigram] += 1
            else:
                unhappyUnigramsNoStopWords[unigram] = 1  
        # Find common bigrams
        if (i > 0):
            bigram = tweet[2][i-1] + ' ' + tweet[2][i]
            if (tweet[0] == 1 or tweet[0] == 2):
                if (bigram in happyBigramsNoStopWords):
                    happyBigramsNoStopWords[bigram] += 1
                else:
                    happyBigramsNoStopWords[bigram] = 1
            if (tweet[0] == 4):
                if (bigram in unhappyBigramsNoStopWords):
                    unhappyBigramsNoStopWords[bigram] += 1
                else:
                    unhappyBigramsNoStopWords[bigram] = 1  
        # Find common trigrams
        if (i > 2):
            trigram = tweet[2][i-2] + ' ' + tweet[2][i-1] + ' ' + tweet[2][i]
            if (tweet[0] == 1 or tweet[0] == 2):
                if (trigram in happyTrigramsNoStopWords):
                    happyTrigramsNoStopWords[trigram] += 1
                else:
                    happyTrigramsNoStopWords[trigram] = 1
            if (tweet[0] == 4):
                if (trigram in unhappyTrigramsNoStopWords):
                    unhappyTrigramsNoStopWords[trigram] += 1
                else:
                    unhappyTrigramsNoStopWords[trigram] = 1  


# Sort word dictionaries by frequency
happyUnigrams = sorted(happyUnigrams.iteritems(), key=lambda (k,v): (v,k), reverse=True)
unhappyUnigrams = sorted(unhappyUnigrams.iteritems(), key=lambda (k,v): (v,k), reverse=True)
happyBigrams = sorted(happyBigrams.iteritems(), key=lambda (k,v): (v,k), reverse=True)
unhappyBigrams = sorted(unhappyBigrams.iteritems(), key=lambda (k,v): (v,k), reverse=True)
happyTrigrams = sorted(happyTrigrams.iteritems(), key=lambda (k,v): (v,k), reverse=True)
unhappyTrigrams = sorted(unhappyTrigrams.iteritems(), key=lambda (k,v): (v,k), reverse=True)

happyUnigramsNoStopWords = sorted(happyUnigramsNoStopWords.iteritems(), key=lambda (k,v): (v,k), reverse=True)
unhappyUnigramsNoStopWords = sorted(unhappyUnigramsNoStopWords.iteritems(), key=lambda (k,v): (v,k), reverse=True)
happyBigramsNoStopWords = sorted(happyBigramsNoStopWords.iteritems(), key=lambda (k,v): (v,k), reverse=True)
unhappyBigramsNoStopWords = sorted(unhappyBigramsNoStopWords.iteritems(), key=lambda (k,v): (v,k), reverse=True)
happyTrigramsNoStopWords = sorted(happyTrigramsNoStopWords.iteritems(), key=lambda (k,v): (v,k), reverse=True)
unhappyTrigramsNoStopWords = sorted(unhappyTrigramsNoStopWords.iteritems(), key=lambda (k,v): (v,k), reverse=True)

# Print dictionaries
print "Most common happy unigrams (including stopwords):"
print happyUnigrams[:30]
print ""
print "Most common unhappy unigrams (including stopwords):"
print unhappyUnigrams[:30]
print ""
print "Most common happy bigrams (including stopwords):"
print happyBigrams[:30]
print ""
print "Most common unhappy bigrams (including stopwords):"
print unhappyBigrams[:30]
print ""
print "Most common happy trigrams (including stopwords):"
print happyTrigrams[:30]
print ""
print "Most common unhappy trigrams (including stopwords):"
print unhappyTrigrams[:30]

print ""

print "Most common happy unigrams (excluding stopwords):"
print happyUnigramsNoStopWords[:30]
print ""
print "Most common unhappy unigrams (excluding stopwords):"
print unhappyUnigramsNoStopWords[:30]
print ""
print "Most common happy bigrams (excluding stopwords):"
print happyBigramsNoStopWords[:30]
print ""
print "Most common unhappy bigrams (excluding stopwords):"
print unhappyBigramsNoStopWords[:30]
print ""
print "Most common happy trigrams (excluding stopwords):"
print happyTrigramsNoStopWords[:30]
print ""
print "Most common unhappy trigrams (excluding stopwords):"
print unhappyTrigramsNoStopWords[:30]

print ""
print "Happy tweets:", labelNumbers["happy"]
print "Unhappy tweets:", labelNumbers["unhappy"]
print "Neutral tweets:", labelNumbers["neutral"]
print "Total tweets:", labelNumbers["total"]

# Utility functions
def containsUnigram(unigram, words):
    if unigram in words:
        return True
    return False

def containsBigram2(bigram, words):
    if not isinstance(bigram, list):
        bigram = bigram.split()
    return containsBigram3(bigram[0], bigram[1], words)

def containsBigram3(word1, word2, words):
    for i in range(1, len(words)):
        if (words[i - 1] == word1 and words[i] == word2):
            return True
    return False

def containsTrigram2(trigram, words):
    if not isinstance(trigram, list):
        trigram = trigram.split()
    return containsTrigram4(trigram[0], trigram[1], trigram[2], words)

def containsTrigram4(word1, word2, word3, words):
    for i in range(2, len(words)):
        if (words[i - 2] == word1 and words[i - 1] == word2 and words[i] == word3):
            return True
    return False


# http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

