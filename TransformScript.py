#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:35:30 2019

@author: marshallgrimmett
"""

#https://blog.myyellowroad.com/using-categorical-data-in-machine-learning-with-python-from-dummy-variables-to-deep-category-66041f734512
#https://blog.myyellowroad.com/using-categorical-data-in-machine-learning-with-python-from-dummy-variables-to-deep-category-42fd0a43b009

import pandas as pd
import numpy as np


def test(df, path):
    df = pd.read_csv(path)
    return df

def run(data, catCols, path):
    data = pd.read_csv(path)
    data.sample(n=10000, replace=True)
    
    # Slow computation
    # Split the location into two seperate columns, longitude and latitude
    
    tempDict = {'Longitude':[], 'Latitude':[]}
    
    for i in range(len(data)):
        try:
            temp = data['Location '][i].split(',')
            long = float(temp[0][1:])
            lat = float(temp[1][:-1])
            
            # Remove outliers
            if long == 0 or lat == 0:
                raise Exception()
            
            tempDict['Longitude'].append(long)
            tempDict['Latitude'].append(lat)
            
        except:
            tempDict['Longitude'].append(np.nan)
            tempDict['Latitude'].append(np.nan)
            
    data.loc[:, 'Longitude'] = tempDict['Longitude']
    data.loc[:, 'Latitude'] = tempDict['Latitude']
    del tempDict
    
    # convert to datetime
    data.loc[:, 'Date Occurred'] = pd.to_datetime(data.loc[:, 'Date Occurred'], format='%m/%d/%Y')
    
    # convert to unix timestamp
    data.loc[:, 'Timestamp'] = data.loc[:, 'Date Occurred'].astype(np.int64) // 10**9
    
    # Removed MO Codes (slow, too many unique combinations), could use word2vec instead
    # Removed 'Address', 'Cross Street' for same reasons
    #http://queirozf.com/entries/one-hot-encoding-a-feature-on-a-pandas-dataframe-an-example
    
#    catCols = [
#        'Area Name', 'Reporting District',
#        'Crime Code Description', 'Victim Age', 'Victim Sex',
#        'Victim Descent', 'Premise Description',
#        'Weapon Description',
#        'Status Description', 'Crime Code 1', 'Crime Code 2', 'Crime Code 3',
#        'Crime Code 4'
#    ]
    
#    catCols = [
#        'Area Name', 'Reporting District', 'Victim Age', 'Victim Sex',
#        'Victim Descent', 'Premise Description',
#        'Status Description', 'Crime Code 1', 'Crime Code 2', 'Crime Code 3',
#        'Crime Code 4'
#    ]
    
    for col in catCols:
        #print('Begin Column:', col)
        #print('\tStarting get_dummies...')
        dummies = pd.get_dummies(data=data[col], prefix=col, drop_first=True , dummy_na=True)
        #print('\tFinished get_dummies...')
        #print('\tStarting concat...')
        data = pd.concat([data, dummies], axis=1)
        #print('\tFinished concat...')
    #print('Done.')
    
    # now drop the original columns
    data.drop(catCols, axis=1, inplace=True)
    
    # Remove other columns that weren't dropped and we don't need.
#    data.drop([
#        'DR Number', 'Date Reported', 'Date Occurred',
#        'Area ID', 'Crime Code',
#        'MO Codes', 'Premise Code',
#        'Weapon Used Code',  'Status Code',
#        'Address', 'Cross Street', 'Location '
#    ], axis=1, inplace=True)
    
    # fill in null values with the mean of each column
    data[['Longitude']] = data[['Longitude']].fillna(data[['Longitude']].mean())
    data[['Latitude']] = data[['Latitude']].fillna(data[['Latitude']].mean())
    
    return data