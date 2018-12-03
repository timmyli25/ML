from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import csv
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


tf.logging.set_verbosity(tf.logging.INFO)

def headlineNN(features, labels, mode):
    '''
    Neural Network for headlines processing.
    '''
    print(features['headlines'].shape)
    input_layer = tf.reshape(features["headlines"],[-1,features["headlines"].shape[1]])
    print(input_layer.shape)
    #embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
    #embeddings = embed(features['headlines'])

    dense1 = tf.layers.dense(inputs=input_layer, units=128, activation=tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense1, units=64, activation=tf.nn.relu)
    dense3 = tf.layers.dense(inputs=dense2, units=1, activation=tf.nn.relu)
    #dense4 = tf.layers.dense(inputs=dense3, units=1, activation=tf.nn.relu)

    flat_dense4 = tf.reshape(dense3, [-1,1])

    #Logits Layer
    logits = tf.layers.dense(inputs=flat_dense4, units=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        #predictions = {'info':features['info'],'logits':logits}
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"logits":logits, 'labels':features['labels']})
    loss = tf.losses.absolute_difference(labels=labels, predictions=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "MAE": tf.metrics.mean_absolute_error(labels=labels, predictions = logits)
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def nltk(headlines):
    stop_words_en = set(stopwords.words('english'))
    wnl = WordNetLemmatizer()

    new_headlines = []
    for headline in headlines:
        words = word_tokenize(headline)
        new_headlines.append(" ".join([wnl.lemmatize(word) for word in words if word not in stop_words_en]))

    return new_headlines

train_df = pd.read_csv('data/ES_train.csv')[['clean_headline','Count','size']].dropna()
test_df = pd.read_csv('data/ES_test.csv')[['clean_headline','Count','size']].dropna()


train_headlines = np.array(train_df['clean_headline'])
train_labels = np.array(train_df['size'],dtype=np.float32)
train_labels = np.reshape(train_labels, (-1,1))
train_count = np.array(train_df['Count'], dtype=np.float32,ndmin=2)
train_headlines = nltk(train_headlines)
vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=1000)


train_headline_binaries =vectorizer.fit_transform(train_headlines).toarray().astype(np.float32)

#train_headline_binaries = np.concatenate((train_headline_binaries, train_count.T), axis=1)

test_headlines = np.array(test_df['clean_headline'])
#passed_headlines = np.array(test_df['clean_headline'])
#test_info = np.array(test_df[['clean_headline','Count','size']])
#test_count = np.array(test_df['Count'], dtype=np.float32, ndmin=2)
test_labels = np.array(test_df['size'],dtype=np.float32)
test_labels = np.reshape(test_labels, (-1,1))
test_headlines = nltk(test_headlines)
test_headline_binaries = vectorizer.transform(test_headlines).toarray().astype(np.float32)

#test_headline_binaries = np.concatenate((test_headline_binaries,test_count.T), axis=1)

headline_estimator = tf.estimator.Estimator(
                    model_fn = headlineNN, model_dir='checkpoints')

tensors_to_log = {}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)


train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {"headlines":train_headline_binaries},
    y=train_labels,
    batch_size=25,
    num_epochs=None,
    shuffle=True)

headline_estimator.train(input_fn=train_input_fn,steps=10000,hooks=[logging_hook])


eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"headlines": test_headline_binaries},
      y=test_labels,
      num_epochs=1,
      shuffle=True)

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={"headlines": test_headline_binaries, 'labels':test_labels},
  num_epochs=1,
  shuffle=False)

predict_results = headline_estimator.predict(input_fn=predict_input_fn,yield_single_examples=True)

eval_results = headline_estimator.evaluate(input_fn=eval_input_fn)


with tf.Session() as sess:
    with open('predictions.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(['actual','prediction'])
        for result in predict_results:
            row = list(result['labels']) + list(result['logits'])
            writer.writerow(row)


print(eval_results)
