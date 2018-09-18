import numpy as np
import tensorflow as tf
import afqstensorutils as atu
from matplotlib import pylab as plt

class Autoencoder(object):
    """
    A class for generating autoencoders.

    This parentclass is a linear map that's useless.
    """
    def __init__(self, size_x, size_q, data):
        self.size_x = size_x
        self.size_q = size_q
        self.dtype = tf.float32
        # Make the trainer
        self.train_step = self._make_train_step(data)
        # Make evaluating graphs
        self.i_x = tf.placeholder( shape=(None,size_x,), dtype=tf.float32 )
        self.o_q = self.encode( self.i_x )
        self.i_q = tf.placeholder( shape=(None,size_q,), dtype=tf.float32 )
        self.o_x = self.decode( self.i_q )
        # Make the loggers for tensorboard
        
    def encode(self, x):
        W = self._var( (self.size_q, self.size_x) )
        b = self._var( (self.size_q,) )
        return tf.matmul(W,x)+b
    def decode(self, q):
        W = self._var( (self.size_x, self.size_q) )
        b = self._var( (self.size_x,) )
        return tf.matmul(W,q)+b
    def goal(self, data):
        pred = self.decode(self.encode(data))
        p = tf.reduce_sum(tf.pow( data - pred, 2) ) 
        return p
    
    def _make_train_step(self, data):
        ts = tf.train.AdamOptimizer(1e-2).minimize(self.goal(data))
        return ts
    def eval_q(self, i_x):
        return sess.eval( self.o_q, feed_dict={self.i_x:i_x} )

    def _var(self, shape, stddev=0.1):
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev),
                               dtype=self.dtype)
    
class PolyAutoencoder(Autoencoder):
    """
    The basic implementation.
    """
    def __init__(self, size_x, size_q, data, Np_enc, Np_dec):
        self.Np_enc = Np_enc
        self.Np_dec = Np_dec
        Autoencoder.__init__(self,size_x, size_q, data)
    def encode(self, x):
        N_coeff = atu.Npolyexpand( self.size_x, self.Np_enc )
        We1 = self._var( (N_coeff, self.size_q) )
        be1 = self._var( (self.size_q,) )
        return tf.matmul( atu.polyexpand(x, self.Np_enc), We1 ) + be1
    def decode(self, q):
        N_coeff = atu.Npolyexpand( self.size_q, self.Np_dec )
        We1 = self._var( (N_coeff, self.size_x) )
        be1 = self._var( (self.size_x,) )
        return tf.matmul( atu.polyexpand(q, self.Np_dec), We1 ) + be1
    
class ClassifyingPolyAutoencoder(Autoencoder):
    """
    This one is like the PolyAutoencoder, but has two layers on the back.

    The idea is that this one learns piecewise branching logic.
    """
    def __init__(self, size_x, size_q, p_enc, p_dec):
        self.Np_enc = Np_enc
        self.Np_dec = Np_dec

