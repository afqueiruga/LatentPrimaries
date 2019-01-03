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


class SaveAtEndHook(tf.train.SessionRunHook):
    def __init__(self,fname):
        self.fname = fname
    def begin(self):
        self._saver = tf.train.Saver()
    def end(self, session):
        # TODO freeze it
        self._saver.save(session, self.fname)
# class DoStuffHook(self)
def train_autoencoder(name, dataset, outerdim, innerdim, hyper=default_hyper):
    autoclass = autoencoder_factory[hyper['type']]
    hyperpath = hyper['type']+'_'+','.join(map(str,hyper['args']))
    training_dir = "training_"+name+"/"+hyperpath
    # Set up the graph from the inputs
    n_epoch = 4000
    image_freq = 200

    graph = tf.Graph()
    with graph.as_default():
        stream = atu.make_datastream(dataset,batch_size=0,buffer_size=1000)
        stream = tf.transpose(stream)
        global_step = tf.train.get_or_create_global_step()
        ae = autoclass(outerdim, innerdim, stream, *hyper['args'])
        init = tf.global_variables_initializer()
        meta_graph_def = tf.train.export_meta_graph(filename=training_dir+"/final_graph.meta")

        # train away
        loghook = tf.train.SummarySaverHook(summary_op=
            tf.summary.scalar("goal",ae.goal),save_steps=50,output_dir=training_dir)
        stophook = tf.train.StopAtStepHook(num_steps=n_epoch)
        saverhook = SaveAtEndHook(training_dir+"/final_variables.ckpt")
        session = tf.train.MonitoredTrainingSession(
            checkpoint_dir=training_dir,
            hooks=[loghook,stophook,saverhook])
        with session as sess:
#             sess.run(init)
            while not sess.should_stop():
                stepnum = global_step.eval(session=sess)
                if stepnum%image_freq==image_freq-1:
                    onum = stepnum/image_freq
                    print(ae.goal.eval(session=sess))
                    header="T, p, rho, h"
                    ae.save_fit(training_dir+"/surf_{0}.csv".format(onum),
                                header,sess=sess)
                sess.run(ae.train_step)
    # Write the graph
    
    
if __name__=="__main__":
    dataset = tf.data.experimental.make_csv_dataset(
        'sharded/*.csv',
        1000,
        select_columns=['T',' p',' rho',' h']
    )
    sets_to_try = [
        #{'type':'Classifying','args':[1,1, 6,12,'tanh']},
        #{'type':'Classifying','args':[1,1, 6,12,'sigmoid']},
        {'type':'Classifying','args':[1,1, 6,12,'relu']}

    ]
    for S in sets_to_try:
        train_autoencoder("water",dataset, 4,2,S)
