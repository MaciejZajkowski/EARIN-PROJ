import pandas as pd
import os


def Preprocessing(sample_size = 10000,df=None):
    
    if df==None:
        df = pd.read_csv('train.csv')
        
    sample = df.groupby('hotel_cluster').apply(lambda x: x.sample(frac=(sample_size/37670293)))
    sample.drop(columns=['user_id'],inplace=True) 
    for column_name in ['date_time','srch_ci','srch_co']:
        sample[column_name + '_month'] = pd.DatetimeIndex(sample[column_name]).month
        sample[column_name +'_day'] = pd.DatetimeIndex(sample[column_name]).day
        sample.drop(columns=[column_name],inplace=True)
    sample.orig_destination_distance = sample.orig_destination_distance.fillna(sample.orig_destination_distance.mean())
    
    sample.index.to_csv(f'../data/datasets/train/sample_{str(sample_size)}.csv')
    return
    sample