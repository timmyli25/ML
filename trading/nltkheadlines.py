import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

headlines = pd.read_csv('ES_master_events.csv').dropna()['Headline']

stop_words_en = set(stopwords.words('english'))
wnl = WordNetLemmatizer()

new_headlines = []
for headline in headlines:
    words = word_tokenize(headline)
    new_headlines.append([wnl.lemmatize(word) for word in words if word not in stop_words_en])
