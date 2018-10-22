from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
import csv

#Code is based on Google's Example of CNNs.

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here




def simpleNN(features, labels, mode):
    '''
    Model function for NN.
    '''
    dense1 = tf.layers.dense(inputs=features["x"],units=64, activation=tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense1,units=64,activation=tf.nn.relu)
    dense3 = tf.layers.dense(inputs=dense2,units=64,activation=tf.nn.relu)
    dense4 = tf.layers.dense(inputs=dense3, units=1)
    flat_dense4 = tf.reshape(dense4, [-1,1])
    dropout = tf.layers.dropout(
        inputs=flat_dense4, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=1)
    predictions = {'id':features['id'],'features': features['x'], 'logits':logits}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    loss = tf.losses.absolute_difference(labels=labels, predictions=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

def main(unused_argv):
  # Load training and eval data
  ALL = ['Id', 'groupId', 'matchId', 'assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'maxPlace', 'numGroups', 'revives',
       'rideDistance', 'roadKills', 'swimDistance', 'teamKills',
       'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints',
       'winPlacePerc']

  variables = ['assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'maxPlace', 'revives',
       'rideDistance', 'swimDistance', 'teamKills',
       'walkDistance', 'weaponsAcquired', 'winPoints']

  testdf = pd.read_csv('test.csv')
  test_data = np.array(testdf[variables], dtype=np.float32)
  test_id = np.array(testdf['Id'], dtype=np.int32)

  test_mean = test_data.mean(axis=0)
  test_std = test_data.std(axis=0)
  test_data = (test_data - test_mean) / test_std

  traindf = pd.read_csv('train.csv')
  train_labels = np.reshape(np.array(traindf['winPlacePerc'],dtype=np.float32),(-1,1))
  train_data = np.array(traindf[variables],dtype=np.float32)
  train_id = np.array(traindf['Id'], dtype=np.int32)

  mean = train_data.mean(axis=0)
  std = train_data.std(axis=0)
  train_data = (train_data - mean) / std
  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=simpleNN, model_dir="checkpoints")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data, 'id':train_id},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)

  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=10000,
      hooks=[logging_hook])


  # Evaluate the model and print results

  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_data, 'id':test_id},
      num_epochs=1,
      shuffle=False)
  predict_results = mnist_classifier.predict(input_fn=predict_input_fn,
                                            yield_single_examples=True)
  #spredict_results = yield(predict_results)
  with open('predictions.csv', 'w') as csvfile:
      writer = csv.writer(csvfile, delimiter = ',')
      writer.writerow(['Id'] + variables + ['winPlacePerc'])
      for result in predict_results:
          row = [result['id']] + list(result['features']) + list(result['logits'])
          writer.writerow(row)

  submit = pd.read_csv('predictions.csv')
  submit = submit[['Id','winPlacePerc']]
  submit.to_csv('sample_submission.csv',index=False)




if __name__ == "__main__":
  tf.app.run()
