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

def split_lower_replace(headline, bad_words, lemmatizer):
    '''
    Turns into lowercase and removes non number of word characters.
    '''
    headline = re.sub("[^0-9A-Za-z]", " ", headline).lower()
    for key in STANDARDIZATION:
        headline = headline.replace(key, STANDARDIZATION[key])

    words = headline.split()
    headline =  " ".join([lemmatizer.lemmatize(word) for word in words if word not in bad_words])
    return headline


def news_pipeline(headlines):
    '''
    Takes an array of headlines and applies the filtering process to standardize it.
    Inputs-
        headlines(np.array)- the collection of headlines
    '''
    bad_words = set(stopwords.words('english'))
    wnl = WordNetLemmatizer()

    headlines = np.array(headlines, dtype = np.unicode_)
    headlines = np.fromiter((split_lower_replace(headline, bad_words, wnl) for headline in headlines), headlines.dtype)

    return headlines

def clean_single_headline(headline, bad_words,lemmatizer):
    headline = split_lower_replace(headline, bad_words, lemmatizer)
    return headline

df = pd.read_csv('ES_master_events.csv')
df['clean_headlines'] = news_pipeline(df['Headline'])
df['event'] = np.where(df['AbsChange'] >= 50,1,0)
X_train, X_test, y_train, y_test = train_test_split(df['clean_headlines'],df['event'])

vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=500)
#vectorizer = TfidfVectorizer()
train_x_binaries = vectorizer.fit_transform(X_train).toarray()
#full_test_x_binaries = vectorizer.transform(full_test_x[:5000]).toarray()
test_x_binaries = vectorizer.transform(X_test).toarray()
clf = BernoulliNB()
clf.fit(train_x_binaries,y_train)
score = clf.score(test_x_binaries, y_test)

def score_headlines(headlines, labels, vectorizer, clf):
    headlines = news_pipeline(headlines)
    binaries = vectorizer.transform(headlines).toarray()
    score = clf.score(binaries, headlines)
    return score

positive_df = df[df['event'] == 1]
positive_headlines = positive_df['Headline']
positive_labels = positive_df['event']

score = score_headlines(positive_headlines, positive_labels, vectorizer, clf)
