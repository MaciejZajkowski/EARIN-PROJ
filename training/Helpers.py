import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from joblib import load

def Preprocessing(data):
    data.drop(columns=['user_id'],inplace=True) 
    for column_name in ['date_time','srch_ci','srch_co']:
        data[column_name + '_month'] = pd.DatetimeIndex(data[column_name]).month
        data[column_name +'_day'] = pd.DatetimeIndex(data[column_name]).day
        data.drop(columns=[column_name],inplace=True)
        
    data['orig_destination_distance'] = data.orig_destination_distance.fillna(1970)
    # sc = load('../parameters/sc_distance.bin')
    # print(len(data.orig_destination_distance))
    # _ = sc.transform(data.orig_destination_distance.values.reshape(-1,1))
    # data['orig_destination_distance'] = _ 
    
    
    destinations = pd.read_csv('../data/original_data/destinations.csv')
    data = data.join(destinations,on='srch_destination_id',how='inner',rsuffix='_right').drop(columns=['srch_destination_id_right','srch_destination_id'])
    x = data.drop(columns=['hotel_cluster'])
    y = data.hotel_cluster
    return x,y
    

def Take_data(sample_size = 10000,df=None):
    #reading train data
    if isinstance(df,type(None)):
        df = pd.read_csv('../data/original_data/train.csv')
        test_id = pd.read_csv('../data/datasets/test/test_id.csv')
        df = df.loc[~df.index.isin(test_id)]
    
    #sampling or reading previosusly sampled data
    if f'sample_{str(sample_size)}.csv' in os.listdir('../data/datasets/train'):
        sample_id = pd.read_csv(os.path.join('../data/datasets/train',f'sample_{str(sample_size)}.csv')).drop(columns=['Unnamed: 0'])
        #FIXME
        sample = df.loc[df.index.isin(sample_id)]
        #print(len(sample_id),df.index.isin(sample_id))
    else:
        sample = df.groupby('hotel_cluster').apply(lambda x: x.sample(frac=(sample_size/37670293))).droplevel(level=0)
        pd.Series(sample.index).to_csv(f'../data/datasets/train/sample_{str(sample_size)}.csv')
    #preprocessing    
    x,y = Preprocessing(data=sample)
    return x,y

def Test_model(model):
    df = pd.read_csv('../data/original_data/train.csv')
    test_id = pd.read_csv('../data/datasets/test/test_id.csv')
    df = df.loc[df.index.isin(test_id)]
    
    x,y = Preprocessing(df)
    predicted = model.predict(x)
    return metrics(predicted,y)
    
def metrics(pred,true):
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,recall_score,balanced_accuracy_score
    metrics = [accuracy_score,balanced_accuracy_score,recall_score,balanced_accuracy_score]
    result = {}
    for metric in metrics:
        result[metric.__name__] = metric(true,pred)
    return result      
    
#print(metrics([1,0,0],[1,1,0]))
