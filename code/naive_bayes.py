import numpy as np
from collections import Counter
from pipeline import *
import itertools
import re
import csv
from sklearn import metrics


# Helper functions
def get_text(tweets, label):
    res = [r[0] for r in tweets if r[1] == label]
    return list(itertools.chain.from_iterable(res))

def count_text(text):
    if text == None:
        return 0
    return Counter(text)

def get_y_count(data, label):
    return len([r for r in data if r[1] == label]) # get the number of a certain label 

# Makes a class prediction based on the class and text
def make_class_prediction(text, counts, class_prob, class_count):
    prediction = 1
    text_counts = Counter(text)
    for word in text:
        prediction *=  text_counts.get(word) * ((counts.get(word, 0) + 1) / float(sum(counts.values()) + class_count))
    return prediction * class_prob

# Makes a class decision for the text
def make_decision(text, make_class_prediction, negative_values, positive_values):
    # Compute the negative and positive probabilities.
    negative_prediction = make_class_prediction(text, negative_values[0], negative_values[1], negative_values[2])
    positive_prediction = make_class_prediction(text, positive_values[0], positive_values[1], positive_values[2])

    # We assign a classification based on which probability is greater.
    if negative_prediction > positive_prediction:
        return -1
    else:
        return 1

#########################################################################

# Runs Naive Bayes on training data specified by tr_data, X, y and tests with test data specificied by test_data, test_labels
def run_test(tr_data, X, y, test_data, test_labels):
    
    data = np.column_stack((X, y))
    negative_text = get_text(data, -1)
    positive_text = get_text(data, 1)

    # Generate word counts for negative tone.
    negative_counts = count_text(negative_text)
    # Generate word counts for positive tone.
    positive_counts = count_text(positive_text)

    # We need these counts to use for smoothing when computing the prediction.
    positive_tweets_count = get_y_count(data, 1)
    negative_tweets_count = get_y_count(data, -1)

    # These are the class probabilities (we saw them in the formula as P(y)).
    prob_positive = positive_tweets_count / float(len(data))
    prob_negative = negative_tweets_count / float(len(data))

    # Make predictions
    negative_values = [negative_counts, prob_negative, negative_tweets_count]
    positive_values = [positive_counts, prob_positive, positive_tweets_count]
    predictions = [make_decision(r[0], make_class_prediction, negative_values, positive_values) for r in test_data]

    # Generate the roc curve using scikits-learn.
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions, pos_label=1)

    # Compute and print evaluation measures
    AUC = metrics.auc(fpr, tpr) # measures the area under the ROC curve.  The closer to 1, the "better" the predictions.
    accuracy = metrics.accuracy_score(y_pred=predictions,y_true=test_labels, normalize=True, sample_weight=None)
    recall = metrics.recall_score(y_pred=predictions,y_true=test_labels, pos_label=1, average='binary')
    precision = metrics.precision_score(y_pred=predictions,y_true=test_labels, pos_label=1, average='binary')
    confusionMatrix = metrics.confusion_matrix(y_pred=predictions,y_true=test_labels, labels=[1,-1], sample_weight=None)
    truePositive = confusionMatrix[0][0]
    trueNegative = confusionMatrix[1][1]
    falsePositive = confusionMatrix[1][0]
    falseNegative = confusionMatrix[0][1]
    specificity = trueNegative/float(trueNegative + falsePositive)
    f_measure = 2*precision*recall/float(precision + recall)
    print "Accuracy:", accuracy
    print "Recall/Sensitivity:", recall
    print "Precision:", precision  
    print "Specificity:", specificity
    print "F-measure", f_measure
    print "Area under ROC curve:", AUC
    print "Confusion Matrix:"
    print confusionMatrix

    return [accuracy, recall, precision, specificity, f_measure, AUC]

#########################################################################

# Runs 10-fold cross-validation and prints results
print ""
print "Naive Bayes 10-Fold Cross-Validation:"
print ""
accuracy = []
recall = []
precision = []
specificity = []
f_measure = []
AUC = []
for i in range(10):
    print "Test", i+1
    results = run_test(tr_data_fold[i], X_fold[i], y_fold[i], test_data_fold[i], test_labels_fold[i])
    accuracy.append(results[0])
    recall.append(results[1])
    precision.append(results[2])
    specificity.append(results[3])
    f_measure.append(results[4])
    AUC.append(results[5])
    print ""

print "Total Results for 10-Fold Cross Validation:"
print "Average accuracy:", sum(accuracy)/10.0
print "Average recall/sensitivity:", sum(recall)/10.0
print "Average precision:", sum(precision)/10.0
print "Average specificity:", sum(specificity)/10.0
print "Average f_measure:", sum(f_measure)/10.0
print "Average area under ROC curve for the tests:", sum(AUC)/10.0
