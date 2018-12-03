import numpy as np
import pandas as pd
import warnings
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.externals import joblib
warnings.filterwarnings(action='ignore', category=UserWarning)


CONTRACT = 'ES'
STANDARDIZATION = {'y y ': 'yy ',
                   'u k ': 'uk ',
                   'u s ': 'us '}


def _clean_headlines(headlines):
    for k in STANDARDIZATION.keys():
        for i in range(len(headlines)):
            if k in headlines[i]:
                headlines[i] = headlines[i].replace(k, STANDARDIZATION[k])
    return headlines


def _clean_words(headlines):
    for headline in headlines:
        if 's' in headline:
            headline.remove('s')
    return headlines


def print_error_types(model, test_headlines_binary, test_y, test_headlines_raw):
    false_positive = 0
    true_negative = 0
    true_positive = 0
    false_negative = 0
    total_true = 0
    total_false = 0
    for i in range(len(test_headlines_binary)):
        headline_binary = [test_headlines_binary[i]]
        raw_headline = test_headlines_raw[i]
        result = test_y[i]
        prediction = model.predict(headline_binary)[0]
        if result == 1 and prediction == 0:
            true_negative += 1
            print('True Negative: {}'.format(raw_headline))
            total_true += 1
        elif result == 0 and prediction == 1:
            false_positive += 1
            #print('False Positive: {}'.format(raw_headline))
            total_false += 1
        elif result == 1 and prediction == 1:
            true_positive += 1
            print('True Positive: {}'.format(raw_headline))
            total_true += 1
        elif result == 0 and prediction == 0:
            false_negative += 1
            total_false += 1
        '''
        print(headline)
        print('Real Value: {} Prediction: {} Probabilities: {}'.format(result,
                                                                       prediction,
                                                                       prediction_proba))
        '''

    print('False Pos: {} True Neg: {} True Pos: {} False Neg: {} Total: {}'.format(false_positive,
                                                                                   true_negative,
                                                                                   true_positive,
                                                                                   false_negative,
                                                                                   len(test_y)))
    true_positive_rate = round(true_positive/total_true, 2)
    false_negative_rate = round(false_negative/total_false, 2)
    print('True Pos Rate: {} False Neg Rate: {}'.format(true_positive_rate, false_negative_rate))


def main():
    df = pd.read_csv('{}_master_events.csv'.format(CONTRACT))
    df = df.dropna()
    df['Event'] = [0] * len(df.index)
    df.loc[df['AbsChange'] >= 150, 'Event'] = 1
    positive_df = df.loc[df.Event == 1]
    negative_df = df.loc[df.Event == 0]

    test_size = 0.25
    test_index_pos = int((1-test_size) * len(positive_df))
    test_index_neg = int((1-test_size) * len(negative_df))
    
    positive_x = np.array(positive_df['Headline'].tolist())
    negative_x = np.array(negative_df['Headline'].tolist())
    positive_y = np.array(positive_df['Event'].tolist())
    negative_y = np.array(negative_df['Event'].tolist())

    test_positive_x = positive_x[test_index_pos:]
    train_positive_x = positive_x[:test_index_pos]
    test_positive_y = positive_y[test_index_pos:]
    train_positive_y = positive_y[:test_index_pos]
    
    test_negative_x = negative_x[test_index_neg:]
    train_negative_x = negative_x[:test_index_neg]
    test_negative_y = negative_y[test_index_neg:]
    train_negative_y = negative_y[:test_index_neg]

    train_positive_x = _clean_headlines(train_positive_x)
    train_negative_x = _clean_headlines(train_negative_x)
    test_positive_x = _clean_headlines(test_positive_x)
    test_negative_x = _clean_headlines(test_negative_x)

    full_train_x = list(train_positive_x) + list(train_negative_x)
    full_train_y = list(train_positive_y) + list(train_negative_y)
    full_test_x = list(test_positive_x) + list(test_negative_x)
    full_test_y = list(test_positive_y) + list(test_negative_y)

    # TODO: Consider using max_features options in the vectorizer. I think they could  help
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=500)
    #vectorizer = TfidfVectorizer()
    vectorizer.fit(full_train_x)
    full_test_x_binaries = vectorizer.transform(full_test_x[:5000]).toarray()
    clf = BernoulliNB()
    #c_weight = int(len(train_negative_y) / (len(full_train_y)))
    #clf = svm.SVC(kernel='linear', class_weight={1: c_weight})
    #clf.fit(full_train_x, full_train_y)
    EPOCHS = 75
    for i in range(EPOCHS):
        batch_train_negative_x = np.random.choice(train_negative_x, len(train_positive_x), replace=False)
        batch_test_negative_x = np.random.choice(test_negative_x, len(test_positive_x), replace=False)
        batch_train_negative_y = [0] * len(batch_train_negative_x)
        batch_test_negative_y = [0] * len(batch_test_negative_x)

        batch_train_x = list(batch_train_negative_x) + list(train_positive_x)
        batch_train_y = list(batch_train_negative_y) + list(train_positive_y)
        batch_test_x = list(test_positive_x) + list(batch_test_negative_x)
        batch_test_y = list(test_positive_y) + list(batch_test_negative_y)

        train_x_binaries = vectorizer.transform(batch_train_x).toarray()
        test_x_binaries = vectorizer.transform(batch_test_x).toarray()
        #clf.fit(train_x_binaries, batch_train_y)
        clf.partial_fit(train_x_binaries, batch_train_y, np.unique(batch_train_y))
        train_score = clf.score(train_x_binaries, batch_train_y)
        test_score = clf.score(test_x_binaries, batch_test_y)
        #test_score = clf.score(full_test_x_binaries[:5000], full_test_y[:5000])
        print('EPOCH: {} had train score {} test score {}'.format(i, train_score, test_score))

    # Save the model
    joblib.dump(clf, '{}_model.joblib'.format(CONTRACT))
    print_error_types(clf, full_test_x_binaries, full_test_y, full_test_x)
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
    fake_headlines_binary = vectorizer.transform(fake_headlines).toarray()
    print(clf.predict(fake_headlines_binary))
    #print(clf.predict_proba(fake_headlines_binary))


if __name__ == '__main__':
    main()