'''
Utilities for for splitting data into a format appropriate for pytext.
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pipeline
import pytext
import sys


def data_split(csvfile,train_outfile,test_outfile, non_event_count,change_col='RealChange',event_col='event',
            headline_col='Headline',time_col='Time',threshold=50,test_ratio=0.25):
    '''
    Splits the csv
    '''
    df = pd.read_csv(csvfile)

    choices = ['down','no_change','up']
    conditions = [
        (df[change_col] <= -(threshold)),
        (df[change_col] >= -(threshold)) & (df[change_col] <= threshold),
        (df[change_col] >= threshold)
    ]

    df[event_col] = np.select(conditions, choices)

    no_change_df = df[df['event'] == 'no_change']

    down_df = df[df['event'] == 'down']
    up_df = df[df['event'] == 'up']

    no_change_df = no_change_df.sample(non_event_count)
    df = pd.concat([down_df, no_change_df, up_df])

    df[headline_col] = pipeline.news_pipeline(df[headline_col])
    df = df[[event_col,headline_col]]

    train, test = train_test_split(df, test_size = test_ratio)
    train.to_csv(train_outfile,sep="\t", index=False,header=False)
    test.to_csv(test_outfile,sep="\t", index=False,header=False)

    return train, test


def find_bad_errors(test_csv,config_file,model_file,columns=['event','headline']):
    '''
    Tests the ability of the model to predict bad errors.
    '''
    config = pytext.load_config(config_file)
    predictor = pytext.create_predictor(config, model_file)

    test_df = pd.read_csv(test_csv,sep='\t',header=None,names=columns).dropna()

    bad_errors = 0
    ok_errors = 0
    good = 0
    count = 0
    correct = 0
    prediction_dic  = {'down':0,'no_change':0,'up':0}
    for row in test_df.iterrows():
        prediction = pytext_predict(predictor, row[1]['headline'])
        prediction_dic[prediction] += 1
        count += 1
        if prediction == 'up' and row[1]['event'] == 'down':
            print(row[1]['headline'], prediction, row[1]['event'])
            bad_errors += 1
        elif prediction == 'down' and row[1]['event'] == 'up':
            print(row[1]['headline'], prediction, row[1]['event'])
            bad_errors += 1
        elif prediction != row[1]['event']:
            ok_errors += 1
        elif prediction == 'up' and row[1]['event'] == 'up':
            good += 1
        elif prediction == 'down' and row[1]['event'] == 'down':
            good += 1

        if prediction == row[1]['event']:
            correct += 1

    print("{} good predictions made.".format(good))
    print("{} bad errors made.".format(bad_errors))
    print("{} ok errors made.".format(ok_errors))
    print("Precision is {}".format(correct/count))
    print(prediction_dic)
    return test_df

def pytext_predict(predictor, headline):
    '''
    Uses the pytext predictor to make  a prediction.
    '''
    result = predictor({"raw_text": headline})
    doc_label_scores_prefix = (
        'scores:' if any(r.startswith('scores:') for r in result)
        else 'doc_scores:'
    )
    best_doc_label = max(
            (label for label in result if label.startswith(doc_label_scores_prefix)),
            key=lambda label: result[label][0],
        # Strip the doc label prefix here
        )[len(doc_label_scores_prefix):]

    return best_doc_label


def main():
    if int(sys.argv[1]) == 0:
        data_split('ES_master_events.csv','train.tsv','test.tsv',40000)

    elif int(sys.argv[1]) == 1:
        find_bad_errors('test.tsv','config/docnn.json','model/ES_simple_model.c2')
if __name__ == '__main__':
    main()
