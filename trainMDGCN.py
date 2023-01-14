# -*- coding: utf-8 -*-
import numpy as np
from funcCNN import *
from GCNModel2 import GCNModel
from BuildSPInst_A import *
import tensorflow as tf
import time



time_start=time.time()
def GCNevaluate(mask1, labels1):
    t_test = time.time()
    outs_val = sess.run([GCNmodel.loss, GCNmodel.accuracy], feed_dict={labels: labels1, mask: mask1})
    return outs_val[0], outs_val[1], (time.time() - t_test)

data_name = 'KSC'
num_classes = 13

learning_rate = 1e-3
epochs=700
img_gyh = data_name+''
img_gt = data_name+'_gt'



Data = load_HSI_data(data_name)
model = GetInst_A(Data['useful_sp_lab'], Data[img_gyh], Data[img_gt], Data['trpos'])
sp_mean = np.array(model.sp_mean, dtype='float32')
sp_label = np.array(model.sp_label, dtype='float32')
trmask = np.matlib.reshape(np.array(model.trmask, dtype='bool'), [np.shape(model.trmask)[0], 1])
temask = np.matlib.reshape(np.array(model.temask, dtype='bool'), [np.shape(model.trmask)[0], 1])
sp_support = []


for A_x in model.sp_A:
    sp_A = np.array(A_x, dtype='float32')
    sp_support.append(np.array(model.CalSupport(sp_A), dtype='float32'))




############################################

mask = tf.placeholder("int32", [None, 1])
labels = tf.placeholder("float", [None, num_classes])

seed=123
np.random.seed(seed)
tf.set_random_seed(seed)
GCNmodel = GCNModel( features = sp_mean, labels = sp_label, learning_rate = learning_rate, 
                    num_classes = num_classes, mask = mask, support = sp_support, scale_num = len(model.sp_A), h = 20 )
sess=tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

for epoch in range(epochs):
    # Training step=
    outs = sess.run([GCNmodel.opt_op, GCNmodel.loss, GCNmodel.accuracy], feed_dict={ labels:sp_label, 
                    mask:trmask })
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]))
        
saver.save(sess, './checkpoints/%s.ckpt'%(data_name), global_step=700)
print("Optimization Finished!")
# Testing
test_cost, test_acc, test_duration = GCNevaluate(temask, sp_label)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
###############
#Pixel-wise accuracy    
outputs = sess.run(GCNmodel.outputs)
pixel_wise_pred = np.argmax(outputs, axis=1) 


# Generating results
pred_mat = AssignLabels(Data['useful_sp_lab'], np.argmax(sp_label, axis=1), pixel_wise_pred, trmask, temask)
scio.savemat('pred_mat.mat',{'pred_mat':pred_mat})
stat_res = GetExcelData(Data[img_gt], pred_mat, Data['trpos'])
scio.savemat('stat_res.mat',{'stat_res':stat_res}) 
