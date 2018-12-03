from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import csv
# Load all files from a directory in a DataFrame.

train_df = pd.read_csv('data/ES_train.csv')[['Headline','Count','size']]
test_df = pd.read_csv('data/ES_test.csv')[['Headline','Count','size']]

# Training input on the whole training set with no limit on training epochs.
train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["size"], num_epochs=None, shuffle=True)

# Prediction on the whole training set.
predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["size"], shuffle=True)
# Prediction on the test set.
predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    test_df, test_df["size"], shuffle=False)

embedded_text_feature_column = hub.text_embedding_column(
    key="Headline",
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

estimator = tf.estimator.DNNRegressor(
    hidden_units=[20,20],
    feature_columns=[embedded_text_feature_column],
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

estimator.train(input_fn=train_input_fn, steps=10000)

train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

print("Training set accuracy: {loss}".format(**train_eval_result))
print("Test set accuracy: {loss}".format(**test_eval_result))

predictions=estimator.predict(input_fn = predict_test_input_fn)
pred = []
for x in predictions:
    pred.append(x['predictions'][0])
