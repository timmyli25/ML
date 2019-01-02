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
    #headline =  " ".join([lemmatizer.lemmatize(word) for word in words if word not in bad_words])
    headline =  " ".join([word for word in words if word not in bad_words])
    return headline


def news_pipeline(headlines):
    '''
    Takes an array of headlines and applies the filtering process to standardize it.
    Inputs-
        headlines(np.array)- the collection of headlines
    Returns-
        headlines(np.array)- collection of headlines with preprocessing applied.
    '''
    bad_words = set(stopwords.words('english'))
    wnl = WordNetLemmatizer()

    headlines = np.array(headlines, dtype = np.unicode_)
    headlines = np.fromiter((split_lower_replace(headline, bad_words, wnl) for headline in headlines), headlines.dtype)

    return headlines

def clean_single_headline(headline, bad_words,lemmatizer):
    headline = split_lower_replace(headline, bad_words, lemmatizer)
    return headline

def score_headlines(headlines, labels, vectorizer, clf):
    headlines = news_pipeline(headlines)
    binaries = vectorizer.transform(headlines).toarray()
    score = clf.score(binaries, headlines)
    return score


def get_predictions(headlines, vectortizer, clf):
    headlines = news_pipeline(headlines)
    binaries = vectorizer.transform(headlines).toarray()
    predictions = clf.predict(binaries)
    return predictions

def show_predictions(headlines,vectorizer, clf, prediction_value):
    for headline in headlines:
        headlinex = news_pipeline([headline])
        binary = vectorizer.transform(headlinex).toarray()
        prediction = clf.predict(binary)[0]
        if prediction == prediction_value:
            print(headline)

def find_bad_errors(headlines, labels, vectorizer, clf):
    '''
    Utilities which will find the headlines which predict the WRONG direction.
    '''
    assert len(headlines) == len(labels), "Length of predictions and labels not the same."
    labels = list(labels)
    clean_headlines = news_pipeline(headlines)
    binaries = vectorizer.transform(clean_headlines).toarray()
    predictions = clf.predict(binaries)
    bad_errors = 0
    ok_errors = 0
    good = 0
    prediction_dic = {-1:0,0:0,1:0}
    for n in range(len(predictions)):
        prediction = predictions[n]
        label = labels[n]
        prediction_dic[prediction] += 1
        if prediction == 1 and label == -1:
            print(headlines.iloc[n], prediction, label)
            bad_errors += 1
        elif prediction == -1 and label == 1:
            print(headlines.iloc[n], prediction, label)
            bad_errors += 1
        elif prediction != label:
            ok_errors += 1
        elif prediction == 1 and label == 1:
            good += 1
        elif prediction == -1 and label == -1:
            good += 1
    print("{} good predictions made.".format(good))
    print("{} bad errors made.".format(bad_errors))
    print("{} ok errors made.".format(ok_errors))
    print(prediction_dic)

def find_good_predictions(headlines, labels, vectorizer, clf):
    '''
    Utilities which will find the headlines which predict the WRONG direction.
    '''
    assert len(headlines) == len(labels), "Length of predictions and labels not the same."
    labels = list(labels)
    clean_headlines = news_pipeline(headlines)
    binaries = vectorizer.transform(clean_headlines).toarray()
    predictions = clf.predict(binaries)
    count = 0
    for n in range(len(predictions)):
        prediction = predictions[n]
        label = labels[n]
        if prediction == 1 and label == 1:
            print(headlines.iloc[n], prediction, label)
            count += 1
        elif prediction == -1 and label == -1:
            print(headlines.iloc[n], prediction, label)
            count += 1
    print("{} Good Predictions made.".format(count))



def test():

    df = pd.read_csv('ES_master_events.csv')
    df['event'] = np.where(df['AbsChange'] >= 50,1,0)

    positive_df = df[df['event'] == 1]
    positive_headlines = positive_df['Headline']
    positive_labels = positive_df['event']

    negative_df = df[df['event'] == 0]
    negative_df = negative_df.sample( positive_df.shape[0])

    total_df = pd.concat([positive_df,negative_df])
    total_df['clean_headlines'] = news_pipeline(total_df['Headline'])

    X_train, X_test, y_train, y_test = train_test_split(total_df['clean_headlines'],total_df['event'])

    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=15000)
    train_x_binaries = vectorizer.fit_transform(X_train).toarray()
    test_x_binaries = vectorizer.transform(X_test).toarray()
    clf = BernoulliNB()
    clf.fit(train_x_binaries,y_train)

    score1 = clf.score(test_x_binaries, y_test)

    x = news_pipeline(positive_headlines)
    x1 = binaries = vectorizer.transform(x).toarray()
    score2 = clf.score(x1, positive_labels)

    print("Total score: {}".format(score1))
    print("Positive score: {}".format(score2))

    fake_headlines = ['uk may says brexit deal will be reached tomorrow',
                      'trump says china deal will be finished',
                      'brexit',
                      'trump fires fed president powell',
                      'fed president powell resigns',
                      'theresa may resigns',
                      'us commerce sec ross us still plans china tariff increase'
                      'matt hazard has decided to lower interest rates',
                      'brennen houge has decided to lower interest rates',
                      'brennen thinks es is going to zero',
                      'little does he know he is wrong',
                      'the pope has been assassinated',
                      'warren buffet says equity markets are overvalued']
    fake_headlines = news_pipeline(fake_headlines)
    fake_headlines_binary = vectorizer.transform(fake_headlines).toarray()
    print(clf.predict(fake_headlines_binary))
