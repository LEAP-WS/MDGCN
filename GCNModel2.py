# -*- coding: utf-8 -*-
import tensorflow as tf
from GCNLayer import *
import numpy as np
from funcCNN import *


def masked_softmax_cross_entropy(preds, labels, mask):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= tf.transpose(mask)
    return tf.reduce_mean(tf.transpose(loss))


def masked_accuracy(preds, labels, mask):
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= tf.transpose(mask)
    return tf.reduce_mean(tf.transpose(accuracy_all))

class GCNModel(object):
    def __init__(self, features, labels, learning_rate, num_classes, mask, support, scale_num, h):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.classlayers = []
        self.labels = labels
        self.inputs = features
        self.scale_num = scale_num
        self.loss = 0
        self.support = support
        self.concat_vec = []
        self.outputs = None
        self.num_classes = num_classes
        self.hidden1 = h 
        self.mask = mask
        self.build()
        
    def _build(self):
        for scale_idx in range(self.scale_num):
            activations = []
            activations.append(self.inputs)
            self.classlayers.append(GraphConvolution(act = tf.nn.softplus,
                                      input_dim = np.shape(self.inputs)[1],
                                      output_dim = self.hidden1,
                                      support = self.support[scale_idx],
                                      bias = True
                                      ))   
            layer = self.classlayers[-1]        
            hidden = layer(activations[-1])
            activations.append(hidden)
                     
            
            support_dynamic = tf.exp(-0.02*tf.matmul(hidden, tf.transpose(hidden)))
            support_dynamic = 0.1*support_dynamic*self.Get01Mat(self.support[scale_idx]) + self.support[scale_idx]#第二层support要改为dynamic
            support_dynamic_1 = tf.matmul(tf.matmul(self.support[scale_idx], support_dynamic), tf.transpose(self.support[scale_idx])) + 0*tf.eye(np.shape(self.support[scale_idx])[0])
            self.classlayers.append(GraphConvolution(act = lambda x:x,
                                      input_dim = self.hidden1,
                                      output_dim = self.num_classes,
                                      support = support_dynamic_1,
                                      bias = True
                                      ))   
            layer = self.classlayers[-1]
            hidden = layer(activations[-1])
            activations.append(hidden)  


            if scale_idx == 0:
                self.concat_vec = activations[-1]
            else:
                self.concat_vec += activations[-1]

            
    def build(self):
        self._build()
        self.outputs = self.concat_vec
        self._loss()
        self._accuracy()
        self.opt_op = self.optimizer.minimize(self.loss)        
        
    def _loss(self):
        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.labels, self.mask)

        
        
    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.labels, self.mask)
    
    def Get01Mat(self, mat1):
        [r, c] = np.shape(mat1)
        mat_01 = np.zeros([r, c])
        pos1 = np.argwhere(mat1!=0)
        mat_01[pos1[:,0], pos1[:,1]] = 1
        return np.array(mat_01, dtype='float32')
