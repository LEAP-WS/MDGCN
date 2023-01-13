import tensorflow as tf
import numpy as np
import time

mask = tf.placeholder("int32", [None, 1])
labels = tf.placeholder("float", [None, 16])

with tf.Session() as sess:
    
    saver = tf.train.import_meta_graph('./checkpoints/IP.ckpt-700.meta')
    # saver.restore(sess,)
    ckpt = tf.train.get_checkpoint_state('./checkpoints/')
    saver.restore(sess, ckpt.model_checkpoint_path)

    def GCNevaluate(mask1, labels1):
        t_test = time.time()
        outs_val = sess.run([],feed_dict={labels: labels1, mask: mask1})
        return outs_val[0], outs_val[1], (time.time() - t_test)

    print(GCNevaluate([[1,1]],[1]))
