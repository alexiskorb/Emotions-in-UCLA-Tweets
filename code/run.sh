#!/bin/bash

python -m pip install numpy
python -m pip install sklearn
python -m pip install cvxopt
python -m pip install nltk


python features.py
python naive_bayes.py
python naive_bayes.py -s
python svm.py -s
