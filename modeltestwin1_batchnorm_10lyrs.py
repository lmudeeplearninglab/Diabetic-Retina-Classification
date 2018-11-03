# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 22:14:21 2018

@author: sixge
"""

import numpy as np
import random
import cv2, os
import tensorflow as tf


class Tensorflow_Model():
    
    __W = None
    __b = None
    
    def __init__(self, image_dims, output_dims):
        self.dims_image = image_dims
        self.dims_output = output_dims
        self.padding = 'SAME'
        self.sess =  tf.Session()
        
        self.__W = {
                1: tf.Variable(tf.truncated_normal([3, 3, 3, 32], stddev=0.1)),
                2: tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1)),
                3: tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1)),
                4: tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1)),
                5: tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1)),
                6: tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1)),
                7: tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1)),
                8: tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1)),
                9: tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.1)),
                10: tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1)),                
                11: tf.Variable(tf.truncated_normal([1*1*512, 1024], stddev=0.1)),
                12: tf.Variable(tf.truncated_normal([1024, output_dims], stddev=0.1)),
            }
        
        self.__b = {
                1: tf.Variable(tf.random_normal([32])),
                2: tf.Variable(tf.random_normal([32])),
                3: tf.Variable(tf.random_normal([64])),
                4: tf.Variable(tf.random_normal([64])),
                5: tf.Variable(tf.random_normal([128])),
                6: tf.Variable(tf.random_normal([128])),
                7: tf.Variable(tf.random_normal([256])),
                8: tf.Variable(tf.random_normal([256])),
                9: tf.Variable(tf.random_normal([512])),
                10: tf.Variable(tf.random_normal([512])),
                
                11: tf.Variable(tf.random_normal([1024 ])),
                12: tf.Variable(tf.random_normal([output_dims])),
            }

    def model(self, inp):
        # Layer 1
        input = inp
        layer1_conv1 = tf.nn.conv2d(input, self.__W[1], strides=[1, 1, 1, 1], padding=self.padding)       
        layer1_relu1 = tf.nn.relu(tf.nn.bias_add(layer1_conv1, self.__b[1]))
        layer1_max_pool1 = tf.nn.max_pool(layer1_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.padding)
        layer1_bn = tf.layers.batch_normalization(inputs = layer1_max_pool1, axis = -1, momentum = 0.9, epsilon=0.001,
                                                  center = True, scale = True, training = True, name = 'layer1_bn')
        w1 = (layer1_max_pool1.get_shape()[1:]).as_list()
#        w21 = (layer1_conv1.get_shape()[1:]).as_list()
        print("w1's shape is", w1)
#        print("w21's shape is", w21)
        
        # Layer 2
        layer2_conv1 = tf.nn.conv2d(layer1_bn, self.__W[2], strides=[1, 1, 1, 1], padding=self.padding)        
        layer2_relu1 = tf.nn.relu(tf.nn.bias_add(layer2_conv1, self.__b[2]))
        layer2_max_pool1 = tf.nn.max_pool(layer2_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.padding)
        layer2_bn = tf.layers.batch_normalization(inputs = layer2_max_pool1, axis = -1, momentum = 0.9, epsilon=0.001,
                                                  center = True, scale = True, training = True, name = 'layer2_bn')
        w2 = (layer2_max_pool1.get_shape()[1:]).as_list()
#        w22 = (layer2_conv1.get_shape()[1:]).as_list()
        print("w2's shape is", w2)
#        print("w22's shape is", w22)
        
        # Layer 3
        layer3_conv1 = tf.nn.conv2d(layer2_bn, self.__W[3], strides=[1, 1, 1, 1], padding=self.padding)        
        layer3_relu1 = tf.nn.relu(tf.nn.bias_add(layer3_conv1, self.__b[3]))
        layer3_max_pool1 = tf.nn.max_pool(layer3_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.padding)
        layer3_bn = tf.layers.batch_normalization(inputs = layer3_max_pool1, axis = -1, momentum = 0.9, epsilon=0.001,
                                                  center = True, scale = True, training = True, name = 'layer3_bn')
        w3 = (layer3_max_pool1.get_shape()[1:]).as_list()
#        w32 = (layer3_conv1.get_shape()[1:]).as_list()
#        #self.__W[4] = tf.Variable(tf.truncated_normal([tf.reduce_prod(self.w4), 1024], stddev=0.1))
        print("w3's shape is", w3)
#        print("w32's shape is", w32)
        
        # Layer 4
        layer4_conv1 = tf.nn.conv2d(layer3_bn, self.__W[4], strides=[1, 1, 1, 1], padding=self.padding)        
        layer4_relu1 = tf.nn.relu(tf.nn.bias_add(layer4_conv1, self.__b[4]))
        layer4_max_pool1 = tf.nn.max_pool(layer4_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.padding)
        layer4_bn = tf.layers.batch_normalization(inputs = layer4_max_pool1, axis = -1, momentum = 0.9, epsilon=0.001,
                                                  center = True, scale = True, training = True, name = 'layer4_bn')
        w4 = (layer4_max_pool1.get_shape()[1:]).as_list()
#        w42 = (layer4_conv1.get_shape()[1:]).as_list()
##       self.__W[4] = tf.Variable(tf.truncated_normal([tf.reduce_prod(self.w4), 1024], stddev=0.1))
        print("w4's shape is", w4)
#        print("w42's shape is", w42)
        
         # Layer 5
        layer5_conv1 = tf.nn.conv2d(layer4_bn, self.__W[5], strides=[1, 1, 1, 1], padding=self.padding)        
        layer5_relu1 = tf.nn.relu(tf.nn.bias_add(layer5_conv1, self.__b[5]))
        layer5_max_pool1 = tf.nn.max_pool(layer5_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.padding)
        layer5_bn = tf.layers.batch_normalization(inputs = layer5_max_pool1, axis = -1, momentum = 0.9, epsilon=0.001,
                                                  center = True, scale = True, training = True, name = 'layer5_bn')
        w5 = (layer5_max_pool1.get_shape()[1:]).as_list()
#        w52 = (layer5_conv1.get_shape()[1:]).as_list()
##       self.__W[4] = tf.Variable(tf.truncated_normal([tf.reduce_prod(self.w4), 1024], stddev=0.1))
        print("w5's shape is", w5)
#        print("w52's shape is", w52)

         # Layer 6
        layer6_conv1 = tf.nn.conv2d(layer5_bn, self.__W[6], strides=[1, 1, 1, 1], padding=self.padding)    
        layer6_relu1 = tf.nn.relu(tf.nn.bias_add(layer6_conv1, self.__b[6]))
        layer6_max_pool1 = tf.nn.max_pool(layer6_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.padding)
        layer6_bn = tf.layers.batch_normalization(inputs = layer6_max_pool1, axis = -1, momentum = 0.9, epsilon=0.001,
                                                  center = True, scale = True, training = True, name = 'layer6_bn')
        w6 = (layer6_max_pool1.get_shape()[1:]).as_list()
        print("w6's shape is", w6)
##        
###        
#         # Layer 7
        layer7_conv1 = tf.nn.conv2d(layer6_bn, self.__W[7], strides=[1, 1, 1, 1], padding=self.padding)       
        layer7_relu1 = tf.nn.relu(tf.nn.bias_add(layer7_conv1, self.__b[7]))
        layer7_max_pool1 = tf.nn.max_pool(layer7_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.padding)
        layer7_bn = tf.layers.batch_normalization(inputs = layer7_max_pool1, axis = -1, momentum = 0.9, epsilon=0.001,
                                                  center = True, scale = True, training = True, name = 'layer7_bn')
        w7 = (layer7_max_pool1.get_shape()[1:]).as_list()
        print("w7's shape is", w7)
        
         # Layer 8
        layer8_conv1 = tf.nn.conv2d(layer7_bn, self.__W[8], strides=[1, 1, 1, 1], padding=self.padding)      
        layer8_relu1 = tf.nn.relu(tf.nn.bias_add(layer8_conv1, self.__b[8]))
        layer8_max_pool1 = tf.nn.max_pool(layer8_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.padding)
        layer8_bn = tf.layers.batch_normalization(inputs = layer8_max_pool1, axis = -1, momentum = 0.9, epsilon=0.001,
                                                  center = True, scale = True, training = True, name = 'layer8_bn')
        w8 = (layer8_max_pool1.get_shape()[1:]).as_list()
        print("w8's shape is", w8)
        
         # Layer 9
        layer9_conv1 = tf.nn.conv2d(layer8_bn, self.__W[9], strides=[1, 1, 1, 1], padding=self.padding)     
        layer9_relu1 = tf.nn.relu(tf.nn.bias_add(layer9_conv1, self.__b[9]))
        layer9_max_pool1 = tf.nn.max_pool(layer9_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.padding)
        layer9_bn = tf.layers.batch_normalization(inputs = layer9_max_pool1, axis = -1, momentum = 0.9, epsilon=0.001,
                                                  center = True, scale = True, training = True, name = 'layer9_bn')
        w9 = (layer9_max_pool1.get_shape()[1:]).as_list()
        print("w9's shape is", w9)
        
         # Layer 10
        layer10_conv1 = tf.nn.conv2d(layer9_bn, self.__W[10], strides=[1, 1, 1, 1], padding=self.padding)
        layer10_relu1 = tf.nn.relu(tf.nn.bias_add(layer10_conv1, self.__b[10]))
        layer10_max_pool1 = tf.nn.max_pool(layer10_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.padding)
        layer10_bn = tf.layers.batch_normalization(inputs = layer10_max_pool1, axis = -1, momentum = 0.9, epsilon=0.001,
                                                  center = True, scale = True, training = True, name = 'layer10_bn')
        w10 = (layer10_max_pool1.get_shape()[1:]).as_list()
        print("w10's shape is", w10)

        # Flatten
        flatten = tf.reshape(layer10_bn, [-1, tf.reduce_prod(w10)]) 
#        wflatten = (flatten.get_shape()).as_list()
#        print("flattern's shape is", wflatten)
    
        # Fully Connected Network
        fc1 = tf.nn.relu(tf.matmul(flatten, self.__W[11]) + self.__b[11])
#        fc1_bn = tf.layers.batch_normalization(inputs = fc1, axis = -1, momentum = 0.9, epsilon=0.001,
#                                                  center = True, scale = True, training = True, name = 'fc1_bn')
#        wfc1 = (fc1.get_shape()).as_list()
#        print("fc1's shape is", wfc1)
        
        
        out = tf.nn.relu(tf.matmul(fc1, self.__W[12]) + self.__b[12])
#        out_bn = tf.layers.batch_normalization(inputs = out, axis = -1, momentum = 0.9, epsilon=0.001,
#                                                  center = True, scale = True, training = True, name = 'out_bn')
#        wfc2 = (out.get_shape()).as_list()
#        print("fc2's shape is", wfc2)
#       print(out.get_shape().as_list())

        return out

#    def one_hot(self, Y):
##        max = np.max(Y)
#        #print(Y)
#        one_hot_encoded = np.zeros([Y.shape[0], 5])
#        for i, y in enumerate(Y):
#            one_hot_encoded[i, y] = 1
#        return one_hot_encoded
#
#
#    def get_x_y(self, data):
##        print('Data shape: {}'.format(data.shape))
##        print('Y values: {}'.format(data[:, 1]))
#        x = np.array([x for x in data[:, 0]]).reshape([-1, self.dims_image['height'], self.dims_image['width'], self.dims_image['channel']])
#        y = self.one_hot(data[:, 1])
##        print('x shape: {}'.format(x.shape))
##        print('y shape: {}'.format(y.shape))
#
#        return x, y


#    def train(self, data):
#        avg_cost = 0
#        with tf.device('/device:GPU:0'):
#        #with tf.device('/cpu:0'):
#            x = tf.placeholder(tf.float32, [None, self.dims_image['height'], self.dims_image['width'], self.dims_image['channel']])
#            y = tf.placeholder(tf.float32, [None, self.dims_output])
#            _y = self.model(x)
#            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_y, labels=y))
#            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
#            corr = tf.equal(tf.argmax(_y, 1), tf.argmax(y, 1))
#            accr = tf.reduce_mean(tf.cast(corr, tf.float32))
#            
#            batch_x, batch_y = self.get_x_y(data)
#            
#            self.sess.run(tf.global_variables_initializer())
#            self.sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
#            avg_cost += self.sess.run(cost, feed_dict={x: batch_x, y: batch_y})/self.dims_output
#            train_acc = self.sess.run(accr, feed_dict={x: batch_x, y: batch_y})
#            print('Average Cost: {}, Training Accuracy: {}'.format(avg_cost, train_acc))


