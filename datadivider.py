# -*- coding: utf-8 -*-
"""
Created on Fri May 25 01:50:40 2018

@author: sixge
"""

import random
import os
import pandas as pd
import numpy as np

class datadivider():
    EXT_TRAIN_DATA = 'train'
    EXT_TEST_DATA = 'test'
    EXT_TRAIN_CSV = 'imageLabels.csv'
    
    def __init__(self):
        self.BASE_PATH = "/Users/sixge/Desktop/data"
        self.TRAIN_PATH = "/Users/sixge/Desktop/data/trainLabels.csv"
        self.TEST_PATH = "/Users/sixge/Desktop/data/testLabels.csv"
        self.Train_Num = 30000
        self.Test_Num = 5122
#        self.BASE_PATH = "/Users/sixge/Google Drive/graduate seminar/Code/data"
#        self.TRAINING_PATH = "/Users/sixge/Google Drive/graduate seminar/Code/data/trainingLabels.csv"
        
    def get_image_name_list(self, path):
        training_csv = pd.read_csv(path)
        headers = training_csv.columns
        return np.array([training_csv[headers[0]], training_csv[headers[1]]])

    def get_image_names(self):
        self.image_names_with_labels = self.get_image_name_list(os.path.join(self.BASE_PATH, self.EXT_TRAIN_CSV)) # returns a tuple
#        self.test_image_names = self.get_image_name_list(os.path.join(self.BASE_PATH, self.EXT_TEST_DATA), 0) # returns just names
#        print('Number of training images: {}\nNumber of testing images: {}'.format(len(self.train_image_names_with_labels[0]), len(self.test_image_names[0])))
#        print(self.train_image_names_with_labels)
        
    def execute(self):
        self.get_image_names()
        self.index = list(range(0, 35122))
        random.shuffle(self.index)
        self.train_image_names_with_labels = [[],[]]
        self.test_image_names_with_labels = [[],[]]
        for i in range(0,self.Train_Num):
            self.train_image_names_with_labels[0].append(self.image_names_with_labels[0][self.index[i]])
            self.train_image_names_with_labels[1].append(self.image_names_with_labels[1][self.index[i]])
        for i in range(self.Train_Num,35122):
            self.test_image_names_with_labels[0].append(self.image_names_with_labels[0][self.index[i]])
            self.test_image_names_with_labels[1].append(self.image_names_with_labels[1][self.index[i]])
            
        train_data = {'image': self.train_image_names_with_labels[0], 'level': self.train_image_names_with_labels[1]}
        df1 = pd.DataFrame(train_data, columns = ['image', 'level'])
        df1.to_csv(self.TRAIN_PATH,index=False)
        
        test_data = {'image': self.test_image_names_with_labels[0], 'level': self.test_image_names_with_labels[1]}
        df2 = pd.DataFrame(test_data, columns = ['image', 'level'])
        df2.to_csv(self.TEST_PATH,index=False)
            
        
datadivider().execute()