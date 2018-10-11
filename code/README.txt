README
Group 9: AXWJ

Environment:
-Our project uses Python 2.7
-Required Libraries:
	*numpy
	*sklearn
	*cvxopt
	*nltk
-Standard Libraries Used:
	*os
	*copy
	*random
	*argparse
	*re
	*csv
	*itertools
	*collections
-The required libraries should be installed by our run.sh script


To Run: 
./run.sh 
	This will run all of our code files.
	It will first find the most common unigrams, bigrams, and trigrams.
	Then, it will perform Naive Bayes with and without stopword exclusion.
	Finally, it will perform SVM, excluding stopwords.
*Our script should download the required libraries, but if this fails, they
might have to be manually downloaded.


Code Files:
-pipeline.py
	This file is a set of supporting code for other files. 
	It takes raw tweet text from tweets.csv and performs pre-processing on them.
	It randomly splits these tweets into ten equally-sized folds for cross-validation
	and combines the data with the appropriate labels.
	Optionally, it will first remove stopwords from tweet text.
-naive_bayes.py
	*To run: 	python naive_bayes.py 		//including stopwords
				python naive_bayes.py -s	//excluding stopwords
	This file runs naive_bayes classification using the training and test data produced
	by pipeline.py. It performs 10-fold cross-validation and also averages the results.
	More information on how this works can be found in our report.
	You can include or exclude stopwords from consideration with the command line option -s.
	Accuracy is similar with or without stopwords.
-svm.py
	*To run: 	python svm.py 		//including stopwords
				python svm.py -s	//excluding stopwords
	This file creates a support vector machine using the training and test data produced
	by pipeline.py. It performs 10-fold cross-validation and also averages the results.
	More information on how this works can be found in our report.
	You can include or exclude stopwords from consideration with the command line option -s.
	The classification does not work very well when stopwords are included.
-features.py
	*To run:	python features.py
	This file takes data from preprocessed_tweets_modified.csv and combines them with the
	corresponding labels from preprocessed_tweets_labels.csv. Then it computes the most
	common unigram, bigrams, and trigrams in each class of tweets (happy or unhappy).
	It does so twice: once including stopwords, and once excluding stopwords.




