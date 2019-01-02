from __future__ import print_function
from autoencoder import *

default_hyper = {
    'type':'Default',
    'args':[],
}
autoencoder_factory = {
    'Default':Autoencoder,
    'Poly':PolyAutoencoder,
    'Deep':DeepPolyAutoencoder,
    'Classifying':ClassifyingPolyAutoencoder,
}
def train_autoencoder(name, dataset, outerdim, innerdim, hyper=default_hyper):
    autoclass = autoencoder_factory[hyper['type']]
    # Set up the graph from the inputs
    graph = tf.Graph()
    with graph.as_default():
        stream = atu.make_datastream(dataset,batch_size=0,buffer_size=1000)
        stream = tf.transpose(stream)
        global_step = tf.train.get_or_create_global_step()
        ae = autoclass(outerdim, innerdim, stream, *hyper['args'])
        init = tf.global_variables_initializer()
    # train away
    n_epoch = 1000
    training_dir = "training_"+name
    with graph.as_default():
        loghook = tf.train.SummarySaverHook(summary_op=
                                            tf.summary.scalar("goal",ae.goal),save_steps=50,output_dir=)
        stophook = tf.train.StopAtStepHook(num_steps=n_epoch)
        session = tf.train.MonitoredTrainingSession(
            checkpoint_dir=training_dir,
            hooks=[loghook,stophook])
        with session as sess:
#             sess.run(init)
            onum = 0
            while not sess.should_stop():
                stepnum = global_step.eval(session=sess)
                if stepnum%(n_epoch/10)==(n_epoch/10)-1:
                    print(ae.goal.eval(session=sess))
                    header="T, p, rho, h"
                    ae.save_fit(training_dir+"/surf_{0}.csv".format(onum),
                                header,sess=sess)
                    onum += 1
                sess.run(ae.train_step)
                
    # Write the graph
    
    
if __name__=="__main__":
    dataset = tf.data.experimental.make_csv_dataset(
        'sharded/*.csv',
        1000,
        select_columns=['T',' p',' rho',' h']
    )
    train_autoencoder("water",dataset, 4,2,
                      {'type':'Classifying','args':[1,2, 6,12]})