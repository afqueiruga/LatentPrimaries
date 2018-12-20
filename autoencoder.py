import numpy as np
import tensorflow as tf
import afqstensorutils as atu
from matplotlib import pylab as plt

def mygrad(y, x):
    yl = tf.unstack(y)
    gl = [ tf.gradients(_,x) for _ in yl ]
    return tf.stack(gl)

class Autoencoder(object):
    """
    A class for generating autoencoders.

    This parentclass is a linear map that's useless.
    """
    def __init__(self, size_x, size_q, data):
        self.size_x = size_x
        self.size_q = size_q
        self.dtype = tf.float32
        # Storage for my variables
        self.vars = {}
        # Make the trainer
        self.data = data
        self.goal = self.make_goal(data)
        self.train_step = self._make_train_step(data)
        # Make evaluating graphs
        self.i_x = tf.placeholder( shape=(None,size_x,), dtype=tf.float32 )
        self.o_q = self.encode( self.i_x )
        self.i_q = tf.placeholder( shape=(None,size_q,), dtype=tf.float32 )
        self.o_x = self.decode( self.i_q )
        self.o_grad_x = tf.gradients(self.o_x, self.i_q)[0]
        # Make the loggers for tensorboard
        
    def encode(self, x):
        W = self._var( "enc:W", (self.size_q, self.size_x) )
        b = self._var( "enc:b", (self.size_q,) )
        return tf.matmul(W,x)+b
    def decode(self, q):
        W = self._var( "dec:W", (self.size_x, self.size_q) )
        b = self._var( "dec:b", (self.size_x,) )
        return tf.matmul(W,q)+b
    def make_goal(self, data):
        pred = self.decode(self.encode(data))
        p = tf.reduce_sum(tf.pow( data - pred, 2) ) 
        return p

    def _make_train_step(self, data):
        ts = tf.train.AdamOptimizer(1e-2).minimize(self.make_goal(data))
        return ts
    def eval_q(self, i_x):
        return sess.eval( self.o_q, feed_dict={self.i_x:i_x} )

    def _var(self, name, shape, stddev=0.1):
        try:
            v = self.vars[name]
        except KeyError:
            v = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev),
                               dtype=self.dtype)
            self.vars[name] = v
        return v
    
    def plot_distance(self, idxs=[0,1]):
        from matplotlib import pylab as plt
        XI,YI = idxs
        inputs = self.data.eval()
        q_enc = self.o_q.eval(feed_dict={self.i_x:inputs})
        x_dec = self.o_x.eval(feed_dict={self.i_q:q_enc})
        plt.plot( inputs[:,XI], inputs[:,YI],',')
        for a,b in zip(inputs,x_dec)[::10]:
            plt.plot( [a[XI],b[XI]], [a[YI],b[YI]],'-k')
        plt.plot(x_dec[:,XI],x_dec[:,YI],'+')
        plt.axis('square');
        
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
        We1 = self._var("enc:W", (N_coeff, self.size_q) )
        be1 = self._var("enc:b", (self.size_q,) )
        return tf.matmul( atu.polyexpand(x, self.Np_enc), We1 ) + be1
    def decode(self, q):
        N_coeff = atu.Npolyexpand( self.size_q, self.Np_dec )
        We1 = self._var("dec:W", (N_coeff, self.size_x) )
        be1 = self._var("dec:b", (self.size_x,) )
        return tf.matmul( atu.polyexpand(q, self.Np_dec), We1 ) + be1
    
class ClassifyingPolyAutoencoder(Autoencoder):
    """
    This one is like the PolyAutoencoder, but has two layers on the back.

    The idea is that this one learns piecewise branching logic.
    """
    def __init__(self, size_x, size_q, p_enc, N_branch, p_branch, p_dec):
        self.Np_enc = Np_enc
        self.Nbranch_dec = N_branch
        self.Np_branch = p_branch
        self.Np_dec = Np_dec
        Autoencoder.__init__(self,size_x, size_q, data)
    def encode(self, x):
        N_coeff = atu.Npolyexpand( self.size_x, self.Np_enc )
        W = self._var("enc:W", (N_coeff, self.size_q) )
        b = self._var("enc:b", (self.size_q,) )
        return tf.matmul( atu.polyexpand(x, self.Np_enc), W ) + b
    def decode(self, q):
        qpoly = atu.polyexpand(q, self.Np_dec)
        N_coeff = atu.Npolyexpand( self.size_q, self.Np_dec )

        
        W1 = self._var("dec:W1", (N_coeff, self.size_q*self.Np_branch) )
        b1 = self._var("dec:b1", (self.size_q*self.Np_branch,) )

        hd = tf.relu(tf.matmul(qpoly,W1)+b1)
        
        W2 = self._var("dec:W2", (N_coeff, self.size_x) )
        b2 = self._var("dec:b2", (self.size_x,) )

        return tf.matmul( ___ ,W2)
    
