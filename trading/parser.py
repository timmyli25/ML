import pandas as pd
import re
from sklearn.utils import shuffle

def regex_headlines(csvfile, trainfile, testfile, index_cut):
    '''
    '''
    df = pd.read_csv(csvfile)
    df['clean_headline'] = df['Headline'].apply(clean_headline)
    df=df.dropna()
    df = shuffle(df)

    train_df = df[:index_cut]
    test_df = df[index_cut:]

    train_df.to_csv(trainfile)
    test_df.to_csv(testfile)

def clean_headline(headline):

    new_headline = " ".join(re.sub("[^0-9A-Za-z]", " ", headline).lower().split())
    return new_headline

regex_headlines('data/ES_top_20_headlines.csv','data/ES_train.csv','data/ES_test.csv',100000)
