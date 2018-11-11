import tensorflow as tf
import scipy.io 
import h5py
import numpy as np
import tensorflow as tf




def conv_layer(input_layer, data, layer_name, strides=[1,1,1,1], padding='VALID'):
    with tf.variable_scope(layer_name):
        W = tf.constant( data[layer_name][layer_name+'_W_1:0'] )
        #b = data[layer_name][layer_name+'_b:0']
        #b = tf.constant( np.reshape(b, (b.shape[0])) )
        x = tf.nn.conv2d(input_layer, filter=W, strides=strides, padding=padding, name=layer_name)
        #x = tf.nn.bias_add(x, b)
        return x


def dense_layer(input_layer, data, layer_name):
    with tf.variable_scope(layer_name):
        W = tf.constant( data[layer_name][layer_name+'_W_1:0'] )
        #b = data[layer_name][layer_name+'_b:0']
        #b = tf.constant( np.reshape(b, (b.shape[0])) )
        x = tf.matmul(input_layer, W)
        #x = tf.nn.bias_add(x, b)
        return x

def batch_norm_layer(input_layer, data, layer_name):
    with tf.variable_scope(layer_name):
        mean = tf.constant( data[layer_name][layer_name+'_running_mean_1:0'] )
        std = tf.constant( data[layer_name][layer_name+'_running_std_1:0'] )
        beta = tf.constant( data[layer_name][layer_name+'_beta_1:0'] )
        gamma = tf.constant( data[layer_name][layer_name+'_gamma_1:0'] )
        return tf.nn.batch_normalization(
            input_layer, mean=mean, variance=std, 
            offset=beta, scale=gamma, 
            variance_epsilon=1e-12, name='batch-norm')


def identity_block(input_layer, stage, data):
    
    with tf.variable_scope('identity_block'):
        
        x = conv_layer(input_layer, data=data, layer_name='res'+stage+'_branch2a')
        x = batch_norm_layer(x, data=data, layer_name='bn'+stage+'_branch2a')
        x = tf.nn.relu(x)
        
        x = conv_layer(x, data=data, layer_name='res'+stage+'_branch2b', padding='SAME')
        x = batch_norm_layer(x, data=data, layer_name='bn'+stage+'_branch2b')
        x = tf.nn.relu(x)
        
        x = conv_layer(x, data=data, layer_name='res'+stage+'_branch2c')
        x = batch_norm_layer(x, data=data, layer_name='bn'+stage+'_branch2c')
        
        x = tf.add(x, input_layer)
        x = tf.nn.relu(x)
        
    return x

def conv_block(input_layer, stage, data, strides=[1, 2, 2, 1]):
    
    with tf.variable_scope('conv_block'):
        
        x = conv_layer(input_layer, data=data, layer_name='res'+stage+'_branch2a', strides=strides)
        x = batch_norm_layer(x, data=data, layer_name='bn'+stage+'_branch2a')
        x = tf.nn.relu(x)
        
        x = conv_layer(x, data=data, layer_name='res'+stage+'_branch2b', padding='SAME')
        x = batch_norm_layer(x, data=data, layer_name='bn'+stage+'_branch2b')
        x = tf.nn.relu(x)
        
        x = conv_layer(x, data=data, layer_name='res'+stage+'_branch2c')
        x = batch_norm_layer(x, data=data, layer_name='bn'+stage+'_branch2c')
        
        shortcut = conv_layer(input_layer, data=data, layer_name='res'+stage+'_branch1', strides=strides)
        shortcut = batch_norm_layer(shortcut, data=data, layer_name='bn'+stage+'_branch1')
        
        x = tf.add(x, shortcut)
        x = tf.nn.relu(x)
        
    return x


class ResNet152(object):

    def __init__(self, resnet152_path):
        tf.reset_default_graph()
        self.RESNET_HEIGHT = 224
        self.RESNET_WIDTH = 224
        self.resnet152_path = resnet152_path

    def build_inputs(self):
        self.images = tf.placeholder(tf.float32, [None, self.RESNET_HEIGHT, self.RESNET_WIDTH, 3], 'images')

    def build_params(self):
        self.data_h5 = h5py.File(self.resnet152_path, 'r')

    def build_model(self):
        with tf.variable_scope('stage1'):
            res = conv_layer(self.images, self.data_h5, 'conv1', strides=[1, 2, 2, 1])
            res = batch_norm_layer(res, self.data_h5, 'bn_conv1')
            res = tf.nn.relu(res)
            res = tf.nn.max_pool(res, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_conv1')
            print 'Stage 1', res.get_shape()
            
    
        with tf.variable_scope('stage2'):
            res = conv_block(input_layer=res, stage='2a', data=self.data_h5, strides=[1, 1, 1, 1])
            res = identity_block(input_layer=res, stage='2b', data=self.data_h5)
            res = identity_block(input_layer=res, stage='2c', data=self.data_h5)
            print 'Stage 2', res.get_shape()

            
        with tf.variable_scope('stage3'):
            res = conv_block(input_layer=res, stage='3a', data=self.data_h5)
            res = identity_block(input_layer=res, stage='3b1', data=self.data_h5)
            res = identity_block(input_layer=res, stage='3b2', data=self.data_h5)
            res = identity_block(input_layer=res, stage='3b3', data=self.data_h5)
            res = identity_block(input_layer=res, stage='3b4', data=self.data_h5)
            res = identity_block(input_layer=res, stage='3b5', data=self.data_h5)
            res = identity_block(input_layer=res, stage='3b6', data=self.data_h5)
            res = identity_block(input_layer=res, stage='3b7', data=self.data_h5)
            print 'Stage 3', res.get_shape()
            
        with tf.variable_scope('stage4'):
            res = conv_block(input_layer=res, stage='4a', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b1', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b2', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b3', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b4', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b5', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b6', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b7', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b8', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b9', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b10', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b11', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b12', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b13', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b14', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b15', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b16', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b17', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b18', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b19', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b20', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b21', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b22', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b23', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b24', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b25', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b26', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b27', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b28', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b29', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b30', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b31', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b32', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b33', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b34', data=self.data_h5)
            res = identity_block(input_layer=res, stage='4b35', data=self.data_h5)

            print 'Stage 4', res.get_shape()
            

            
            
        with tf.variable_scope('stage5'):
            res = conv_block(input_layer=res, stage='5a', data=self.data_h5)
            res = identity_block(input_layer=res, stage='5b', data=self.data_h5)
            res = identity_block(input_layer=res, stage='5c', data=self.data_h5)
            print 'Stage 5', res.get_shape()
            # self.features = res
            self.features = tf.reshape(res, [-1, 49, 2048])
                    

    def build(self):
        self.build_inputs()
        self.build_params()
        self.build_model()












