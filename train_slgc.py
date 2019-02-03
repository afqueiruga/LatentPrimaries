from __future__ import print_function
import tensorflow as tf
from autoencoder_trainer import train_autoencoder

training_dir = "/Users/afq/Google Drive/networks/"
data_dir = "/Users/afq/Documents/Dropbox/ML/primaryautoencoder/"
if __name__=="__main__":
    n_epoch = 100
    dataset = tf.data.experimental.make_csv_dataset(
        data_dir+'data_files/water_iapw_logp_sharded/*.csv',
        5000,
        select_columns=['T','p','rho','h'],
        column_defaults=[tf.float64,tf.float64,tf.float64,tf.float64]
    )
    sets_to_try = [
        {'type':'Deep','args':[1,[],1,[12,12,12]]},
        
         {'type':'Deep','args':[2,[12],3,[12,12,12]]},
         {'type':'Deep','args':[1,[12],1,[12,12,12]]},
         {'type':'Poly','args':[2,7]},
         {'type':'Poly','args':[3,7]},
         {'type':'Classifying','args':[1,1, 6,12,'sigmoid']},
        {'type':'Classifying','args':[2,1, 6,12,'sigmoid']},
        {'type':'Classifying','args':[1,1, 12,24,'sigmoid']},
        {'type':'Classifying','args':[1,5, 6,12,'sigmoid']},
        {'type':'Classifying','args':[2,1, 12,24,'sigmoid']},
        {'type':'Classifying','args':[2,3, 12,24,'sigmoid']},
        {'type':'Classifying','args':[2,4, 12,24,'sigmoid']},
        {'type':'Classifying','args':[2,5, 12,24,'sigmoid']},
        {'type':'Classifying','args':[2,6, 18,36,'sigmoid']},
        {'type':'Classifying','args':[2,6, 24,48,'sigmoid']},
    ]
    for S in sets_to_try:
        train_autoencoder("water_slgc_logp_64",dataset, 4,2,S,
                         training_dir=training_dir,
                         n_epoch=n_epoch, image_freq = 1500)
        