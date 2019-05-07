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
        
class DoStuffHook(tf.train.SessionRunHook):
    def __init__(self, freq=500):
        self.freq = freq
        self.global_step = tf.train.get_or_create_global_step()
    def after_run(self,run_context,run_values):
        stepnum = run_context.session.run(self.global_step)
        if stepnum%self.freq==self.freq-1:
            self.func(run_context,run_values)
    def __call__(self, func):
        self.func = func
        return self
    
def train_autoencoder(name, dataset, outerdim, innerdim, hyper=default_hyper,
                     training_dir='',n_epoch=5000, image_freq=1500):
    autoclass = autoencoder_factory[hyper['type']]
    def sanitize(x):
        return str(x).replace(' ','').replace('[','(').replace(']',')')
    hyperpath = hyper['type']+'_'+','.join(map(sanitize,hyper['args']))
    training_dir = training_dir+"/training_"+name+"/"+hyperpath
    
    graph = tf.Graph()
    with graph.as_default():
        # Set up the graph from the inputs
        stream = atu.make_datastream(dataset,batch_size=0,buffer_size=1000)
        stream = tf.transpose(stream)
        global_step = tf.train.get_or_create_global_step()
        onum = tf.Variable(0,name="csv_output_num")
        ae = autoclass(outerdim, innerdim, stream, *hyper['args'])
        init = tf.global_variables_initializer()
        meta_graph_def = tf.train.export_meta_graph(filename=training_dir+"/final_graph.meta")
        
        # Add some more hooks
        loghook = tf.train.SummarySaverHook(
            summary_op=tf.summary.scalar("goal",ae.goal),
            save_steps=50,output_dir=training_dir)
        stophook = tf.train.StopAtStepHook(num_steps=n_epoch)
        saverhook = SaveAtEndHook(training_dir+"/final_variables")
        # Make a closure into a hook
        @DoStuffHook(freq=image_freq)
        def extrahook(ctx,run_values):
            stepnum = global_step.eval(session=ctx.session)
            onum_val = onum.eval(session=ctx.session)
            onum_val+=1
            onum.load(onum_val, session=ctx.session)
            print(ae.goal.eval(session=ctx.session))
            header="T, p, rho, h"
            ae.save_fit(training_dir+"/surf_{0}.csv".format(onum_val),
                        header,sess=ctx.session)
        # set up the session
        session = tf.train.MonitoredTrainingSession(
            checkpoint_dir=training_dir,
            hooks=[loghook,stophook,saverhook,extrahook])
        # train away
        with session as sess:
            while not sess.should_stop():
                sess.run(ae.train_step)
    
    
if __name__=="__main__":
    training_dir = "/Users/afq/Google Drive/networks/"
    data_dir = "/Users/afq/Dropbox/ML/primaryautoencoder/data_files/"
    data_dir = "/Users/afq/Documents/Dropbox/ML/primaryautoencoder/data_files/"
    dataset = tf.data.experimental.make_csv_dataset(
        data_dir+'water_lg_sharded/*.csv',
        5000,
        select_columns=['T',' p',' rho',' h'],
        column_defaults=[tf.float64,tf.float64,tf.float64,tf.float64]
    )
    sets_to_try = [
#         {'type':'Classifying','args':[1,1, 6,12,'tanh']},
#         {'type':'Classifying','args':[1,1, 6,12,'sigmoid']},
#         {'type':'Classifying','args':[1,3, 6,12,'tanh']},
#         {'type':'Classifying','args':[1,3, 6,12,'sigmoid']},
#         {'type':'Classifying','args':[1,4, 6,12,'sigmoid']},
#         {'type':'Classifying','args':[1,5, 6,12,'sigmoid']},
#         {'type':'Classifying','args':[1,1, 6,12,'relu']},
#         {'type':'Deep','args':[1,[],1,[8,8,8]]},
#         {'type':'Deep','args':[1,[],2,[8,8,8]]},
#         {'type':'Deep','args':[1,[],3,[8,8,8]]},
        {'type':'Poly','args':[1,5]},
#         {'type':'Poly','args':[1,6]},
#         {'type':'Poly','args':[1,7]},
    ]
    initialization_modes = [
        "pT",
        "rhoh",
        "rand",
    ]
    inner_penalty = [
        0.0,
        0.1,
    ]
#     n_epoch = 25000
    n_epoch = 10
#     import joblib
#     @joblib.delayed
#     def job(S):
#         train_autoencoder("water_lg",dataset, 4,2,S,
#                           training_dir=training_dir,
#                           n_epoch = n_epoch)
#         return 1
#     joblib.Parallel(n_jobs=2)(job(S) for S in sets_to_try)
    for S in sets_to_try:
        train_autoencoder("water_lg",dataset, 4,2,S,
                          training_dir=training_dir,
                          n_epoch = n_epoch)
