from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np


train_df = pd.read_csv('data/ES_train.csv')[['clean_headline','Count','size']].dropna()[:50000]
test_df = pd.read_csv('data/ES_test.csv')[['clean_headline','Count','size']].dropna()[:20000]

train_headlines = np.array(train_df['clean_headline'])
train_labels = np.array(train_df['size'],dtype=np.float32)
train_labels = np.reshape(train_labels, (-1,1))
train_headlines = np.array([headline.split() for headline in train_headlines])

MLB = MultiLabelBinarizer()
train_headline_binaries =np.array(MLB.fit_transform(train_headlines), dtype=np.float32)

test_headlines = np.array(test_df['clean_headline'])
test_labels = np.array(test_df['size'],dtype=np.float32)
test_labels = np.reshape(test_labels, (-1,1))
test_headlines = np.array([headline.split() for headline in test_headlines])
test_headline_binaries = np.array(MLB.transform(test_headlines), dtype=np.float32)
