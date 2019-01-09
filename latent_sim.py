import numpy as np
import tensorflow as tf
import afqstensorutils as atu

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
        saver = tf.train.import_meta_graph(location+'/final_graph.meta')
        saver.restore(self._sess,location+'final_variables')
        # hooks to the decoder
        self.i_q = graph.get_tensor_by_name('i_q:0') 
        self.o_x = graph.get_tensor_by_name('decode:0')
        # hooks to the encoder
        self.i_x = graph.get_tensor_by_name('i_x:0') 
        self.o_q = graph.get_tensor_by_name('encode:0')
        
        scale = np.loadtxt(scale_file,skiprows=1,delimiter=',')
        def unshift(x, scale):
            return (scale[2,:]-scale[1,:])*x + scale[0,:]
        def shift(x,scale):
            return (x-scale[0,:])/(scale[2,:]-scale[1,:])
        
    def build_dae(self):
        pass
    
    def initialize(self, x):
        pass