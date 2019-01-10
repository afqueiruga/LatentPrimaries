import numpy as np
import tensorflow as tf
import afqstensorutils as atu

def unshift(x, scale):
    return (scale[2,:]-scale[1,:])*x + scale[0,:]
def shift(x,scale):
    return (x-scale[0,:])/(scale[2,:]-scale[1,:])

class LatentSim():
    """Object for storing a simulation state"""
    def __init__(self):
        "Constructor"
        self._graph = None
        self._sess = None
        
    def load_model(self, location, scale_file="", logp=True):
        """Read the autoencoder from a tensorflow file"""
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)
#         with self._sess as sess:
        with self._graph.as_default():
            saver = tf.train.import_meta_graph(location+'/final_graph.meta')
        saver.restore(self._sess,location+'/final_variables')
        # hooks to the decoder
        self.i_q = self._graph.get_tensor_by_name('i_q:0') 
        self.o_x = self._graph.get_tensor_by_name('decode:0')
        # hooks to the encoder
        self.i_x = self._graph.get_tensor_by_name('i_x:0') 
        self.o_q = self._graph.get_tensor_by_name('encode:0')
        # The simulation inputs and outputs in the unscaled space
        # (the prior conditioning of the data put it on [-1,1])
        if scale_file:
            self.logp = logp
            self.scale = np.loadtxt(scale_file,skiprows=1,delimiter=',')
            unshifted = unshift(self.o_x,self.scale[:,0:4])
            if logp:
                self.o_s = tf.concat([ tf.expand_dims(unshifted[:,0], -1),
                                      tf.expand_dims(tf.math.exp(unshifted[:,1]), -1),
                                      unshifted[:,2:]], axis=1)
            else:
                shifted = self.i_s
                self.o_s = unshifted
        else:
            self.scale = None
            self.o_s = self.o_x
        
    def decode(self, q):
        " Go from q to s"
        
        return self._sess.run(self.o_s, feed_dict={self.i_q:q})
    
    def encode(self, s):
        " From a q that satisfies s "
        if self.scale:
            if self.logp:
                slog = s.copy()
                slog[:,1] = np.log(slog[:,1])
                s=slog
            x = shift(s, self.scale[0:4])
        else:
            x = s
        return _self._sess.run(self.o_q, feed_dict={self.i_x:x})
        
    def build_dae(self):
        pass
    
    def initialize(self, s):
        pass