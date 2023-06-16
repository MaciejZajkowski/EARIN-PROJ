import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from joblib import load
import tensorflow as tf
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
import sys
import numpy as np
import random
import pathlib
from tensorflow.keras import layers

def Preprocessing(data):
    data.drop(columns=['user_id'],inplace=True) 
    for column_name in ['date_time','srch_ci','srch_co']:
        data[column_name]= pd.to_datetime(data[column_name], errors = 'coerce')
        data[column_name + '_month'] = pd.DatetimeIndex(data[column_name]).month
        data[column_name +'_day'] = pd.DatetimeIndex(data[column_name]).day
        data.drop(columns=[column_name],inplace=True)
    
    data['orig_destination_distance'] = data.orig_destination_distance.fillna(1970)
    sc = load('../parameters/sc_distance.bin')
    #print(len(data.orig_destination_distance))
    _ = sc.transform(data.orig_destination_distance.values.reshape(-1,1))
    data['orig_destination_distance'] = _ 
    
    data.dropna(inplace=True)
    
    destinations = pd.read_csv('../data/original_data/destinations.csv')
    data = data.join(destinations,on='srch_destination_id',how='inner',rsuffix='_right').drop(columns=['srch_destination_id_right','srch_destination_id'])

    x = data.drop(columns=['hotel_cluster'])
    y = data.hotel_cluster
    #print(len(x),len(y),len(data))
    return x,y
    

def Take_data(sample_size = 10000,df=None):
    #reading train data
    if isinstance(df,type(None)):
        df = pd.read_csv('../data/original_data/train.csv')
       # test_id = pd.read_csv('../data/datasets/test/test_id.csv')
        #df = df.loc[~df.index.isin(test_id)]
    
    #sampling or reading previosusly sampled data
    if f'sample_{str(sample_size)}.csv' in os.listdir('../data/datasets/train'):
        sample_id = pd.read_csv(os.path.join('../data/datasets/train',f'sample_{str(sample_size)}.csv')).drop(columns=['Unnamed: 0'])['0']
        #FIXME
        sample = df.loc[sample_id]
        #print(len(sample_id),df.index.isin(sample_id))
    else:
        sample = df.groupby('hotel_cluster').apply(lambda x: x.sample(frac=(sample_size/37670293))).droplevel(level=0)
        pd.Series(sample.index).to_csv(f'../data/datasets/train/sample_{str(sample_size)}.csv')
    #preprocessing    
    x,y = Preprocessing(data=sample)
    return x,y

def Test_model(model,df=None,sample_size=50000):
    if isinstance(df,type(None)):
        df = pd.read_csv('../data/original_data/train.csv')
    test_id = pd.read_csv('../data/datasets/test/test_id.csv')['0'].sample(sample_size)
    df = df.loc[test_id]
    
    x,y = Preprocessing(df)
    predicted = model.predict(x)
    return metrics(predicted,y)
    
def metrics(pred,true):
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,recall_score,f1_score
    metrics = [accuracy_score,balanced_accuracy_score,balanced_accuracy_score,recall_score,f1_score]
    result = {}
    for metric in metrics:
        if metric.__name__ in ['recall_score','f1_score']:
            result[metric.__name__] = metric(true,pred,average='macro')    
        else:
            result[metric.__name__] = metric(true,pred)
    return result 
    
#print(metrics([1,0,0],[1,1,0]))

def make_model_nn(params:dict):
    kernel_regularizer = regularizers.l2(1e-5)
    bias_regularizer = regularizers.l2(1e-5)
    model = tf.keras.Sequential()
    tf.keras.layers.Input(173),
    for key in params:
        if 'dense' in key:
            model.add(tf.keras.layers.Dense(params[key], activation = 'relu',
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer))
            
            if 'bn' in params.keys() and params['bn'] ==1:
                model.add(tf.keras.layers.BatchNormalization())
        if 'drop' in key and params[key] != 0:
            model.add(tf.keras.layers.Dropout(params[key]))
        else:
            pass
    model.add(tf.keras.layers.Dense(100, activation = 'softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model

def load_nn_data(sample,val_frac = 0.3):
    X=  pd.read_pickle('../data/datasets/train/nn/X_train.pkl')
    y = pd.read_pickle('../data/datasets/train/nn/y_train.pkl')
    if sample < len(X):
        _ = y.to_frame().groupby('hotel_cluster').apply(lambda x: x.sample(frac=(sample/len(y)))).droplevel(level=0)
        X = X.loc[X.index.isin(_.index)]
        y = _.values
    return train_test_split(X,y, test_size=val_frac)


def take_params(posibilities,num=None,proc=None):
    n_dict_list = []
    for iteration, k in enumerate(posibilities):
        if iteration == 0:
            for val in posibilities[k]:
                n_dict ={k:val}
                n_dict_list.append(n_dict)
        else:
            temp = []
            for val in posibilities[k]:
                n_dict_list_cpy = n_dict_list.copy()
                for ndict in n_dict_list:
                    _ = ndict.copy()
                    _[k] = val
                    n_dict_list_cpy.append(_)
                temp = temp + n_dict_list_cpy
            n_dict_list = [ x for x in temp if len(x) == iteration+1]
    if proc is not None:
        if proc > 1 or proc < 0:
            return n_dict_list
        num = int(proc * len(n_dict_list))        
    
    if num is not None:
        return random.sample(n_dict_list,num)
    else:
        return n_dict_list
    
def test_nn_model(model):
    X=  pd.read_pickle('../data/datasets/train/nn/X_test.pkl')
    y = pd.read_pickle('../data/datasets/train/nn/y_test.pkl')
    pred = model.predict(X,verbose = 0)
    pred = np.argmax(pred,axis =1)
    return metrics(pred,y)

def test_cv_model(model):
    print('tutaj')
    desktop = pathlib.Path('../../pics_test/')
    ids = [ int(x.name.split('_')[1]) for x in  list(desktop.rglob("*.png"))]
    y = pd.read_pickle('../data/datasets/train/coding_pic.pkl')
    y = y.loc[y.index.isin(ids)]
    test_ds = tf.keras.preprocessing.image_dataset_from_directory('../../pics_test/', image_size=(30,30),batch_size=128)
    pred = model.predict(test_ds,verbose = 0)
    pred = np.argmax(pred,axis =1)
    return metrics(pred,y)

def load_cv_data(sample,proc=0.3):
    if sample > 20000:
        train_ds,val_ds = tf.keras.preprocessing.image_dataset_from_directory('../../pics_dataset/' , image_size=(30,30),batch_size=128,validation_split = proc,subset = 'both',seed = 42)
    else:
        train_ds,val_ds = tf.keras.preprocessing.image_dataset_from_directory(f'../../pics_sample_{sample}/' , image_size=(30,30),batch_size=128,validation_split = proc,subset = 'both',seed = 42)
    return train_ds,val_ds
            
def make_model_cv(params):
    model = tf.keras.models.Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(5, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(10, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten()])
    
    kernel_regularizer = regularizers.l2(1e-5)
    bias_regularizer = regularizers.l2(1e-5)
    for key in params:
        if 'dense' in key:
            model.add(tf.keras.layers.Dense(params[key], activation = 'relu',
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer))
            
            if 'bn' in params.keys() and params['bn'] ==1:
                model.add(tf.keras.layers.BatchNormalization())
        if 'drop' in key and params[key] != 0:
            model.add(tf.keras.layers.Dropout(params[key]))
        else:
            pass
    model.add(tf.keras.layers.Dense(100, activation = 'softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model
