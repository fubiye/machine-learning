# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 21:27:18 2019

@author: biyef
"""
from PIL import Image, ImageFilter
import tensorflow as tf
import matplotlib.pyplot as plt
import mnist_lenet5_backward
import mnist_lenet5_forward
import numpy  as np

def imageprepare():
    im = Image.open('D:/workspace/machine-learning/mnist/img/origin-9.png')
    plt.imshow(im)
    plt.show()
    #print(type(im.getdata()))
    tv = list(im.getdata()) 
    tva = [(255-x)*1.0/255.0 for x in tv] 
    #return np.asarray(im)
    return tva

result=imageprepare()

#x = tf.placeholder(tf.float32, [None, 784])
#x = result
with tf.Graph().as_default() as g: 
    x = tf.placeholder(tf.float32,[1,
	mnist_lenet5_forward.IMAGE_SIZE,
	mnist_lenet5_forward.IMAGE_SIZE,
	mnist_lenet5_forward.NUM_CHANNELS]) 
    #x = tf.placeholder(tf.float32, [None, 784])
    #ipt = imageprepare()
    #y_ = tf.placeholder(tf.float32, [None, mnist_lenet5_forward.OUTPUT_NODE])
    #y = mnist_lenet5_forward.forward(x,False,None)
#	x = tf.placeholder(tf.float32,[
#            [ipt],
#            mnist_lenet5_forward.IMAGE_SIZE,
#            mnist_lenet5_forward.IMAGE_SIZE,
#            mnist_lenet5_forward.NUM_CHANNELS]) 
#    y_ = tf.placeholder(tf.float32, [None, mnist_lenet5_forward.OUTPUT_NODE])
#    y = mnist_lenet5_forward.forward(x,False,None)	
#    
#    ema = tf.train.ExponentialMovingAverage(mnist_lenet5_backward.MOVING_AVERAGE_DECAY)
#    ema_restore = ema.variables_to_restore()
#    saver = tf.train.Saver(ema_restore)
#		
#    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) 
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
    #image = tf.image.decode_png('D:/workspace/machine-learning/mnist/img/origin-2.png')
    
   # image = tf.cast(image, tf.float32)
    
    y_conv = mnist_lenet5_forward.forward(x,False,None)	
    #eva = mnist_lenet5_forward.forward([image],False,None)
    #prediction = tf.argmax(y,1)
    saver = tf.train.Saver() 
    with tf.Session(graph=g) as sess:
        init_op = tf.global_variables_initializer() 
        sess.run(init_op)
        ckpt = tf.train.get_checkpoint_state(mnist_lenet5_backward.MODEL_SAVE_PATH)
        saver.restore(sess, ckpt.model_checkpoint_path)
        reshaped_xs = np.reshape([result],(  
		    1,
        	mnist_lenet5_forward.IMAGE_SIZE,
        	mnist_lenet5_forward.IMAGE_SIZE,
        	mnist_lenet5_forward.NUM_CHANNELS))
#        reshaped_x = np.reshape([ipt],(
#                    [ipt],
#        	        mnist_lenet5_forward.IMAGE_SIZE,
#        	        mnist_lenet5_forward.IMAGE_SIZE,
#        	        mnist_lenet5_forward.NUM_CHANNELS))
#                    accuracy_score = sess.run(accuracy, feed_dict={x:reshaped_x,y_:[2]}) 
        prediction=tf.argmax(y_conv,1)
        predint=prediction.eval(feed_dict={x: reshaped_xs}, session=sess)
        print('recognize result:')
        print(predint[0])