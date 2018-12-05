import pipeline
import pandas as pd
import numpy as np
import warnings
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
warnings.filterwarnings(action='ignore', category=UserWarning)

STANDARDIZATION = {'y y ': 'yy ',
                   'u k ': 'uk ',
                   'u s ': 'us '}

bad_words = set(stopwords.words('english'))
wnl = WordNetLemmatizer()

df = pd.read_csv('ES_master_events.csv')

threshold = 50
non_event_count = 5000

choices = [-1,0,1]
conditions = [
    (df['RealChange'] <= -(threshold)),
    (df['RealChange'] >= -(threshold)) & (df['RealChange'] <= threshold),
    (df['RealChange'] >= threshold)
]

df['event'] = np.select(conditions, choices)

no_change_df = df[df['event'] == 0]
no_headlines = no_change_df['Headline']
no_label = no_change_df['event']

down_df = df[df['event'] == -1]
down_headlines = down_df['Headline']
down_label = down_df['event']

up_df = df[df['event'] == 1]
up_headlines = up_df['Headline']
up_label = up_df['event']

no_change_df = no_change_df.sample(non_event_count)

total_df = pd.concat([down_df, no_change_df, up_df])
total_df['clean_headlines'] = pipeline.news_pipeline(total_df['Headline'])

X_train, X_test, y_train, y_test = train_test_split(total_df['clean_headlines'],total_df['event'])

vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=3000)
train_x_binaries = vectorizer.fit_transform(X_train).toarray()
test_x_binaries = vectorizer.transform(X_test).toarray()
clf = BernoulliNB()
clf.fit(train_x_binaries,y_train)
score1 = clf.score(test_x_binaries, y_test)
print("Total Score: {}".format(score1))

x = pipeline.news_pipeline(down_headlines)
x1 = binaries = vectorizer.transform(x).toarray()
score2 = clf.score(x1, down_label)

print("Negative Score: {}".format(score2))
'''
x = pipeline.news_pipeline(no_headlines)
x1 = binaries = vectorizer.transform(x).toarray()
score3 = clf.score(x1, no_label)
print("No Score: {}".format(score3))
'''
x = pipeline.news_pipeline(up_headlines)
x1 = binaries = vectorizer.transform(x).toarray()
score4 = clf.score(x1, up_label)
print("Up Score: {}".format(score4))
