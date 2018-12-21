from __future__ import print_function
import unittest as ut
from autoencoder import *

class AutoencoderTest(ut.TestCase):
    def test_linear(self):
        """The default autoencoder should be able to learn a 
        linear mapping"""
        q = np.linspace(0,1,1000)
        xy = np.array([[2,3]]).T * q
        print(xy)
        graph = tf.Graph()
        with graph.as_default():
            stream = atu.make_datastream(
                dataset,batch_size=1000,buffer_size=1000)
            tr_x = stream #tf.transpose(stream)
            #au = Autoencoder(2,1,tr_x)
            au = PolyAutoencoder(2,1,tr_x, 1,1)
            init=tf.global_variables_initializer()
            
    def test_quadratic(self):
        """The polynomial autoencoder should be able to learn 
        a polynomial surface"""
