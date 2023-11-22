''''
Class to handle the data processing needed for the NN
'''

import pandas as pd
import numpy as np

class Processing():

    def __init__(self):
        
        self.Test = pd.DataFrame()
        self.Train = pd.DataFrame()

    def normalise(series):
        '''Normalise a series of data'''
        return(series - (series.min()*0.8)) / ((series.max()*1.2) - (series.min()*0.8))
    
    def split_data(self,df,split=0.7):
        '''Split the data accordinly'''

        n = df.shape[0]
        train_n = int(n*split)
        
        df_train = df.iloc[:train_n,:]
        df_test = df.iloc[train_n:,:]

        return df_train,df_test
    
    def train_test_split(self, df):
        '''split the data up into train & test sets'''

        df_MF   = df[df['Class_Move-Forward'] == 1]
        df_SIRT = df[df['Class_Slight-Right-Turn'] == 1]
        df_SHRT = df[df['Class_Sharp-Right-Turn'] == 1]
        df_SILT = df[df['Class_Slight-Left-Turn'] ==1]
        
        train_data_MF,test_data_MF     = self.split_data(df_MF)
        train_data_SIRT,test_data_SIRT = self.split_data(df_SIRT)
        train_data_SHRT,test_data_SHRT = self.split_data(df_SHRT)
        train_data_SILT,test_data_SILT = self.split_data(df_SILT)
    
        self.Test = pd.concat([test_data_MF,test_data_SIRT,test_data_SHRT,test_data_SILT])
        self.Train = pd.concat([train_data_MF,train_data_SIRT,train_data_SHRT,train_data_SILT])

        data_test = np.array(self.Test).T
        data_train = np.array(self.Train).T

        y_test = data_test[24:28]
        X_test = data_test[0:24]

        y_train = data_train[24:28]
        X_train = data_train[0:24]

        X_train = X_train.astype(float)
        X_test = X_test.astype(float)   

        return y_test, y_train, X_test, X_train