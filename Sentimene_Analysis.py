# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 16:05:12 2018

@author: rohit
"""
# import all packages
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import unicodedata
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import random
from sklearn.metrics import f1_score
import pickle
from sklearn.linear_model import SGDClassifier
import string
from nltk.tokenize import WhitespaceTokenizer
from sklearn.metrics.classification import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB


# load the excel files 
df = pd.read_csv(r'../Sentiment.csv')
df_lab = df 

# convert data frames to lists
X = []
X_token = []
Y = []

# convert the label df into list
for index, row in df_lab.iterrows():
    Y.append(row['sentiment'])

del index
del row

for index, row in enumerate(Y):
    if 'negative' in Y[index].lower():
        Y[index] = 1
    else:
        Y[index] = 0
        
del index
del row      

# convert data df into list of strings striping the leading and trailing spaces
for index, row in df.iterrows():
    X.append((unicodedata.normalize('NFKD', row['text']).encode('utf-8','ignore')).decode('utf-8').strip())
    
del index
del row    
del df
del df_lab  
  
# importing stop words and using Regex tokenizer to remove punctuations  
stopWords = set(stopwords.words('english'))
#hindi_stop_words = pd.read_csv(r"G:\IUPUI\MS Project\stopwords_hindi.txt", sep="\n", header=None, encoding='utf-8')#.apply(set)
#marathi_stop_words = pd.read_csv(r"G:\IUPUI\MS Project\stopwords-mr-master\stopwords_marathi.txt", sep="\n", header=None, encoding='utf-8')#.apply(set)

#tokenizer = RegexpTokenizer(r'\w+')

for row in X:
    #remove punctuations
    exclude = set(string.punctuation)
    row = ''.join(ch for ch in row if ch not in exclude)
    #words = tokenizer.tokenize(row.lower())
    words = WhitespaceTokenizer().tokenize(row.lower())

    wordsFiltered = []
    
    for w in words:
         if (w not in stopWords):
             wordsFiltered.append(w)
            
    wordsFiltered = ' '.join(wordsFiltered)            
            
    X_token.append(wordsFiltered)            

del w
del wordsFiltered
del row
del words

for idx, row in enumerate(X_token):
    if row == '':
        X_token.remove(X_token[idx])
        Y.remove(Y[idx])
        
del idx
del row   

vectorizer = CountVectorizer()

X_bagOfWords = vectorizer.fit_transform(X_token)

X_train, X_test, y_train, y_test = train_test_split(X_bagOfWords, Y, test_size=0.25, random_state=0)

#print('SGD Classification')
#sgd_clf = SGDClassifier(loss="log", class_weight="balanced")
#sgd_clf.fit(X_train, y_train)
#
#y_pred_sgd = sgd_clf.predict(X_test)
#print('Accuracy:', accuracy_score(y_test, y_pred_sgd))
#print('F1 score:', f1_score(y_test, y_pred_sgd, average='macro'))
#print(classification_report(y_test, y_pred_sgd, digits=4))

print('MNB Classification')
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

y_pred_mnb = mnb.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred_mnb))
print('F1 score:', f1_score(y_test, y_pred_mnb, average='macro'))
print(classification_report(y_test, y_pred_mnb, digits=4))



