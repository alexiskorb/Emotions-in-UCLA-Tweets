import os
import csv
import numpy as np
from pipeline import *
from numpy import array
from numpy import linalg
from cvxopt import matrix,solvers
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn import metrics

stopWords = list(stopwords.words("english"))

class SVM(object):

    def __init__(self):
        self.alphas = []
        self.sup_x = []
        self.sup_y = []
        self.b = 0.0
        self.w = []

    def fit(self, x, y):
    
        # C is the cost for soft margin
        C = 1.0
        n = array(x).shape[0]
        p = array(x).shape[1]
        
        # Solve Dual form:
        # min_alpha Q = 1/2 * sum(sum(a_i*a_j*y_i*y_j*x_iTx_j)) - sum(a_i)
        # subject to:
        # sum(a_i*y_i) = 0
        # -a_i <= 0
        # a_i <= C (optional)
        
        # P = yyT * K
        P_array = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                P_array[i][j] = y[i] * y[j] * np.dot(x[i], x[j])
        P = matrix(P_array)
        
        # q = - [1, 1, 1, 1, ...]T
        q = matrix(np.ones(n) * -1)

        # G is diagonal matrix of -1 combined with diagonal matrix of 1
        G_array = np.zeros((2 * n, n))
        for i in range(n):
            G_array[i][i] = -1
            G_array[i+n][i] = 1
        G = matrix(G_array)
        
        # h is [0, 0, 0, ...., 1, 1, 1, 1, ...]
        h = matrix(np.concatenate((np.zeros(n), np.ones(n) * C)))

        # A = [y_1, y_2, y_3 ...]
        A = matrix(array(y), (1, n))
        
        # b = 0
        b = matrix(0.0)
        
        # Solve QP problem with cvxopt
        sol = solvers.qp(P, q, G, h, A, b)
        
        # Get alphas
        alphas = np.array(sol['x']).reshape(-1)

        # If alpha_i is non-zero then x_i is a support vector
        # Use 1e-5 to avoid rounding error
        for i in range(n):
            if alphas[i] > 1e-5:
                self.alphas.append(alphas[i])
                self.sup_x.append(x[i])
                self.sup_y.append(y[i])
                
        # Calculate weight
        self.weight = [0] * p
        for i in range(len(self.alphas)):
            for j in range(p):
                self.weight[j] += self.alphas[i] * self.sup_y[i] * self.sup_x[i][j]
                
        # Calculate bias
        for i in range(len(self.alphas)):
            if self.alphas[i] < C - 1e-5:
                self.b += self.sup_y[i] - np.dot(array(self.weight), array(self.sup_x[i]))
        self.b /= len(self.alphas)
                
    def predict(self, x):
        return np.sign(np.dot(array(x), array(self.weight)) + array(self.b))

def run_test(tr_data, X, y, test_data, test_labels):
    # find all non-neutral tweets
    test_X = [a[0] for a in test_data]
    XWithMood = []
    yWithMood = []
    testWithMood = []
    labelWithMood = []
    tokensUsed = 30
    for i in range(len(X)):
        if y[i] is 1 or y[i] is -1:
            XWithMood.append(X[i])
            yWithMood.append(float(y[i]))
    for i in range(len(test_X)):
        if test_labels[i] is 1 or test_labels[i] is -1:
            testWithMood.append(test_X[i])
            labelWithMood.append(test_labels[i])
    # find most frequent unigrams (including exclamation mark) in X as tokens
    happyUnigrams = {}
    unhappyUnigrams = {}
    for i in range(len(XWithMood)):
        for j in range(len(XWithMood[i])):
            unigram = XWithMood[i][j]
            if (yWithMood[i] == 1):
                if (unigram in happyUnigrams):
                    happyUnigrams[unigram] += 1.0
                else:
                    happyUnigrams[unigram] = 1.0
            elif (yWithMood[i] == -1):
                if (unigram in unhappyUnigrams):
                    unhappyUnigrams[unigram] += 1.0
                else:
                    unhappyUnigrams[unigram] = 1.0
    happyUnigrams = sorted(happyUnigrams.iteritems(), key=lambda (k,v): (v,k), reverse=True)
    happyUnigrams = happyUnigrams[:tokensUsed]
    unhappyUnigrams = sorted(unhappyUnigrams.iteritems(), key=lambda (k,v): (v,k), reverse=True)
    unhappyUnigrams = unhappyUnigrams[:tokensUsed]
        
    # count happy and unhappy unigrams in each tweet
    count_X = [[0.0 for j in range(2)] for i in range(len(XWithMood))] 
    count_test = [[0.0 for j in range(2)] for i in range(len(testWithMood))] 
    for i in range(len(XWithMood)):
        for j in range(len(XWithMood[i])):
            unigram = XWithMood[i][j]
            for k in range(tokensUsed):
                if unigram == happyUnigrams[k][0]:
                    count_X[i][0] += 1.0
                if unigram == unhappyUnigrams[k][0]:
                    count_X[i][1] += 1.0
    for i in range(len(testWithMood)):
        for j in range(len(testWithMood[i])):
            unigram = testWithMood[i][j]
            for k in range(tokensUsed):
                if unigram == happyUnigrams[k][0]:
                    count_test[i][0] += 1.0
                if unigram == unhappyUnigrams[k][0]:
                    count_test[i][1] += 1.0
    # apply linear SVM
    clf = SVM()
    clf.fit(count_X, yWithMood)
    
    # predict test_data
    predictions = [clf.predict(x) for x in count_test]
    
    # Generate the roc curve using scikits-learn.
    fpr, tpr, thresholds = metrics.roc_curve(labelWithMood, predictions, pos_label=1)

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
print "SVM 10-Fold Cross-Validation:"
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
