import pandas as pd
import os

def Preprocessing(data):
    data.drop(columns=['user_id'],inplace=True) 
    for column_name in ['date_time','srch_ci','srch_co']:
        data[column_name + '_month'] = pd.DatetimeIndex(data[column_name]).month
        data[column_name +'_day'] = pd.DatetimeIndex(data[column_name]).day
        data.drop(columns=[column_name],inplace=True)
    data.orig_destination_distance = data.orig_destination_distance.fillna(data.orig_destination_distance.mean())
    return data
    

def Take_data(sample_size = 10000,df=None):
    #reading train data
    if df==None:
        df = pd.read_csv('../data/original_data/train.csv')
        test_id = pd.read_csv('../data/datasets/test/test_id.csv')
        df = df.loc[~df.index.isin(test_id)]
    
    #sampling or reading previosusly sampled data
    if f'sample_{str(sample_size)}.csv' in os.listdir('../data/datasets/train'):
        sample_id = pd.read_csv(os.path.join('../data/datasets/train'f'sample_{str(sample_size)}.csv'))
        sample = df.loc[df.index.isin(sample_id)]
    else:
        sample = df.groupby('hotel_cluster').apply(lambda x: x.sample(frac=(sample_size/37670293)))
    
    #preprocessing    
    sample = Preprocessing(data=sample)
    
    sample.index.to_csv(f'../data/datasets/train/sample_{str(sample_size)}.csv')
    
    return sample

def Test_model(model):
    df = pd.read_csv('../data/original_data/train.csv')
    test_id = pd.read_csv('../data/datasets/test/test_id.csv')
    df = df.loc[df.index.isin(test_id)]
    
    df = Preprocessing(df)
    model.predict(df)
    
def metrics(pred,true):
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,recall_score,balanced_accuracy_score
    metrics = [accuracy_score,balanced_accuracy_score,recall_score,balanced_accuracy_score]
    result = {}
    for metric in metrics:
        result[metric.__name__] = metric(true,pred)
    return result    
    
print(metrics([1,0,0],[1,1,0]))
