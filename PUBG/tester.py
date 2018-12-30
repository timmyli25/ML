import pandas as pd
import numpy as np

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
