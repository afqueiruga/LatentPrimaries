from __future__ import print_function
import tensorflow as tf
from autoencoder_trainer import train_autoencoder

training_dir = "/Users/afq/Google Drive/networks/"
data_dir = "/Users/afq/Documents/Dropbox/ML/primaryautoencoder/data_files/"
#data_dir = "/Users/afq/Dropbox/ML/primaryautoencoder/data_files"
if __name__=="__main__":
    n_epoch = 5000
#     dataset = tf.data.experimental.make_csv_dataset(
#         data_dir+'data_files/water_linear_sharded/*.csv',
#         500,
#         select_columns=['T','p','rho','h'],
#         column_defaults=[tf.float64,tf.float64,tf.float64,tf.float64]
#     )
    dataset = 'water_linear_scaled'
    sets_to_try = [
        {'type':'Poly','args':[1,1],'ini':'pT','cae':0},
    ]
    for S in sets_to_try:
        train_autoencoder("water_linear",dataset, 4,2,S,
                         training_dir=training_dir,
                          data_dir=data_dir,
                          
                         n_epoch=n_epoch)
        