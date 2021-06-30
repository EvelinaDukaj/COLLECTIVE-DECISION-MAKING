# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 08:59:16 2020

@author: PC_Nt
"""
# 

import pickle 
import pandas as pd
import csv

pickle_in = open("\Archives\{0}_{1}_{2}_{3}.pkl","rb")
example_dict = pickle.load(pickle_in)
print(example_dict)

with open('feed_in_dataframe.pkl', 'rb') as f:
    data = pickle.load(f)

df = pd.DataFrame.from_dict(data=example_dict)
df.to_csv("CentralDB_50terminals0.05_0.05_0.05_0.5.csv")

# do not use this until necessary

df = pd.read_csv('CentralDB_50terminals0.05_0.05_0.05_0.05.csv')
for id in range(0,10):
    df_id = df[df['id'] == id]
    file_name = str(id) + '.csv'
    df_id.to_csv(file_name)