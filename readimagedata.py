# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:25:12 2018

@author: sixge
"""

import numpy as np
import cv2, os
import pandas as pd
from random import randint
#import tensorflow as tf
#from PIL import Image
import pprint, pickle

class readimage():
    EXT_TRAIN_DATA = 'train'
    EXT_TEST_DATA = 'test'
    EXT_IMAGE_CSV = 'imageLabels.csv'
    EXT_TRAIN_CSV = 'trainLabels.csv'
    EXT_TEST_CSV = 'testLabels.csv'
    
    IMAGE_WIDTH = 512 #1536
    IMAGE_HEIGHT = 340 #1024
    N_CHANNELS = 3

    
    #argv = "/Users/sixge/Google Drive/graduate seminar/Code/data"
    
    def __init__(self, GENERATOR_BATCH_SIZE, filename):
       # self.argv = 
#        self.BASE_PATH = "/Users/sixge/Google Drive/graduate seminar/Code/data"
        self.BASE_PATH = "/Users/sixge/Desktop/data"
        #self.BASE_PATH = "/Users/deyi/Google Drive/graduate seminar/Code/data"
        self.dims_image = {'width': self.IMAGE_WIDTH, 'height': self.IMAGE_HEIGHT, 'channel': self.N_CHANNELS}
        self.dims_output = 5
        self.GENERATOR_BATCH_SIZE = GENERATOR_BATCH_SIZE
        self.EXT_CSV = filename
    
    def get_daved_mean(self):
        pkl_file = open('data.pkl', 'rb')
        self.mean = pickle.load(pkl_file)    
         
    def get_image_name_list(self, path):
        training_csv = pd.read_csv(path)
        headers = training_csv.columns
        return np.array([training_csv[headers[0]], training_csv[headers[1]]])

    def get_image_names(self):
        self.train_image_names_with_labels = self.get_image_name_list(os.path.join(self.BASE_PATH, self.EXT_CSV)) # returns a tuple
#        self.test_image_names = self.get_image_name_list(os.path.join(self.BASE_PATH, self.EXT_TEST_DATA), 0) # returns just names
#        print('Number of training images: {}\nNumber of testing images: {}'.format(len(self.train_image_names_with_labels[0]), len(self.test_image_names[0])))
#        print(self.train_image_names_with_labels)
        
    def image_transformation(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
#        print("111", img.shape)
        img = np.asarray(img)
        img = img - self.mean
#        print("Shape is", img.shape)
#        img = img.tolist()

        
        #print(image_path)
#        width, height, channel = img.shape
#        print("image's width and height are", width, height)
#        try:
#        img = cv2.resize(img, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
#        out_image = cv2.normalize(img.astype(np.float32), out_image, alpha=-0.5, beta=0.5,\
#        print("Origional size is ", img.shape)
#        img2 = tf.image.per_image_standardization(img)
#        print("New size is ", img2.shape)
#        except:
#            print("This image does not work")
        #new_image_path=image_path[:-5]+'resize_'+image_path[-5:]
        #print(new_image_path)
        #cv2.imwrite( new_image_path,img)
        #cv2.imshow('image',img)
        return img.reshape((self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.N_CHANNELS))   #?
        
    
    def image_batch_generator(self, array, batch_size, ext):
          path = os.path.join(self.BASE_PATH, ext)
          indIm = list()
          
          for n in range(0, batch_size):
              indIm.append(randint(0,len(array[0])-1))
#          print("randon index of batch items are", indIm)
#          print("length of data is", len(array[0]))
          data_batch = []
        
          for i in range(0, batch_size):              
              batch = array[0][indIm[i]]                   
#              print(array[0][indIm[i]])
#              print("The batch is", batch)              
              image_path = '{}.jpeg'.format(os.path.join(path, batch))         
              #The reading output is saved with three divisions: image file, label, and its name
              data_batch.append((self.image_transformation(image_path), array[1][indIm[i]], batch))
          
          yield(np.array(data_batch))
          
    def execute(self):
        self.get_image_names()
        self.get_daved_mean()
        training_batch_generator = self.image_batch_generator(self.train_image_names_with_labels, self.GENERATOR_BATCH_SIZE, self.EXT_TRAIN_DATA)
        return training_batch_generator
    
    
#        batch_total10 = [0, 0, 0]
#        mean10 = [0, 0, 0]
#        batch_total100 = [0, 0, 0]
#        mean100 = [0, 0, 0]
#        total_mean = [0, 0, 0]
#        path = os.path.join(self.BASE_PATH, self.EXT_TRAIN_DATA)
#        for i in range(30000):          
#            batch = self.train_image_names_with_labels[0][i]
#            image_path = '{}.jpeg'.format(os.path.join(path, batch))         
#            img = cv2.imread(image_path)
#            img = cv2.resize(img, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
################################################################################            
###The following code calculate the overal 3D means for all images           
#            
#            img = np.asarray(img)
#            batch_total10 = batch_total10 + img
##            print(img)
#            if (i+1) % 10 == 0:
#                mean10 = batch_total10 / 10
#                batch_total100 = batch_total100 + mean10
#                batch_total10 = [0, 0, 0]
##                print("mean10 is", mean10)
##                print(mean10)
#            if (i+1) % 100 == 0:               
#                mean100 = batch_total100 / 10
#                total_mean = total_mean + mean100
#                print("The epach", ((i+1) - (i+1) % 100))
#                batch_total100 = [0, 0, 0]
#        mean = total_mean / 300
#        
#        #save the Overall 3D mean into data.pkl
#        output = open('data.pkl', 'wb')
#        pickle.dump(mean, output)
#        output.close()
##        print("The shape is", mean.shape())
#        print("The total mean is", mean)
#        
###########################################################################
#The following code subtracts mean from each images with respect to 3D
#        path2 = os.path.join(self.BASE_PATH, self.EXT_TRAIN_nDATA)
#        for i in range(10):
#            batch = self.train_image_names_with_labels[0][i]
#            image_path = '{}.jpeg'.format(os.path.join(path, batch))
#            img = cv2.imread(image_path)
#            img = np.asarray(img)
#            mean_totalarray = np.asarray(mean_total)
#            img2 = img - mean_totalarray
##            print(img)
#            im = Image.fromarray(img)
#            image_path2 = '{}.jpeg'.format(os.path.join(path2, batch))
#            im.save(image_path2)
#            
        
############################################################################      
#The following code tests if the image is valid
#                try:
#                    img = cv2.resize(img,(512,340))
#                print ("Image",batch,"works")
#                except:
#                    print ("Image",batch,"is invalid")
                
         
#readimage(10, 'trainLabels.csv').execute()


    