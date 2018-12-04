import csv
import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt
from glob import glob
from tqdm import tqdm
import re



CONTRACT = 'ES'
VOLUME_CUTOFF = 100
COLS = ['Time',
        'Headline',
        'HotLevel',
        'Ticker',
        'Topic',
        'Person',
        'Transact_time',
        'price',
        'size',
        'Agressor Side']


class HeadlineEvent(object):
    def __init__(self, ts, headline, df):
        self.ts = ts
        self.headline = headline
        self.df = df.copy()
        self.abs_change = -1
        self.real_change = -1
        self.range = -1
        self.volume = -1
        self.initial_calc()

    def initial_calc(self):
        self.df.loc[:, 'Transact_time'] = pd.to_datetime(self.df['Transact_time'], format='%Y-%m-%d %H:%M:%S.%f')
        self.df.loc[:, 'Time'] = pd.to_datetime(self.df['Time'], format='%Y-%m-%d %H:%M:%S.%f')
        self.df.loc[:, 'time_delta'] = (self.df['Transact_time'] - self.df['Time']) / pd.Timedelta('1s')
        self.df = self.df.loc[self.df['time_delta'] > 0]
        self.df = self.df.loc[self.df['time_delta'] < 5]
        if len(self.df.index) <= 1:
            return
        self.abs_change = abs(self.df['price'].iloc[-1] - self.df['price'].iloc[0])
        self.real_change = self.df['price'].iloc[-1] - self.df['price'].iloc[0]
        self.range = self.df['price'].max() - self.df['price'].min()
        self.volume = self.df['size'].sum()


def clean_headline(headline):
    #headline = headline.remove("'")
    #new_headline = " ".join(re.sub("[^0-9A-Za-z]", "", headline).lower().split())
    return headline


def main():
    df_list = []
    for fname in sorted(glob('headline_prints/*.csv')):
        if CONTRACT in fname:
            part_df = pd.read_csv(fname)
            part_df = part_df.loc[part_df['Agressor Side'] != 'No Aggressor']
            df_list.append(part_df)


    master_df = pd.concat(df_list)
    events = []
    with open('{}_master_events.csv'.format(CONTRACT), 'w', newline='') as f:
        header = ['Time',
                  'Headline',
                  'AbsChange',
                  'Range',
                  'RealChange',
                  'Volume']
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
        for headline, partial_df in master_df.groupby(['Time', 'Headline']):
            cleaned_headline = clean_headline(headline[1])
            event = HeadlineEvent(headline[0], cleaned_headline, partial_df)
            events.append(event)
            line = [headline[0],
                    headline[1],
                    event.abs_change,
                    event.range,
                    event.real_change,
                    event.volume]
            csv_writer.writerow(line)

    sorted_events = sorted(events, key=lambda i: i.range)
    sorted_events.reverse()
    top_5 = int(len(events) / 20)
    i = 0
    sorted_events[0].df.to_csv('wtf.csv')
    for e in sorted_events:
        if e.volume <= VOLUME_CUTOFF:
            i += 1
            continue
        buy_df = None
        sell_df = None
        for side, partial_df in e.df.groupby('Agressor Side'):
            if side == 'Buy':
                buy_df = partial_df
            elif side == 'Sell':
                sell_df = partial_df
        print('{} had range {}'.format(e.headline, e.range))
        """
        if buy_df is not None:
            plt.scatter(buy_df['time_delta'], buy_df['price'], buy_df['size'], c='g')
        if sell_df is not None:
            plt.scatter(sell_df['time_delta'], sell_df['price'], sell_df['size'], c='r')
        if buy_df is not None or sell_df is not None:
            plt.title(e.headline)
            plt.show()
        """
        i += 1
        if i >= top_5:
            break


if __name__ == '__main__':
    main()
