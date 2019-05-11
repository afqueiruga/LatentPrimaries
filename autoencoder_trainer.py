from __future__ import print_function
from autoencoder import *

default_hyper = {
    'type':'Default',
    'args':[],
    'ini':'rand',
    'cae':0,
}

def sanitize_string(x):
    return str(x).replace(' ','').replace('[','(').replace(']',')')
def string_identifier(hyper):
    return hyper['type'] + '_' + \
           hyper['ini']  + '_' + str(hyper['cae']) +'_'+ \
           ','.join(map(sanitize_string,hyper['args']))


def AutoencoderFactory(hyper, outerdim, innerdim, stream_mini, stream_all):
    "Parse the hyperparameter dict"
    class_factory = {
        'Default':Autoencoder,
        'Poly':PolyAutoencoder,
        'Deep':DeepPolyAutoencoder,
        'Classifying':ClassifyingPolyAutoencoder,
    }
    autoclass = class_factory[hyper['type']]
    return autoclass(outerdim, innerdim, stream_mini,
                     *hyper['args'],
                     data_all=stream_all,
                     encoder_init=hyper['ini'],
                     cae_lambda=hyper['cae'])


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
    
def train_autoencoder(name, dataname, outerdim, innerdim, 
                      hyper=default_hyper,
                      training_dir='',data_dir='',n_epoch=5000, image_freq=1500):
    
    hyperpath = string_identifier(hyper)
    training_dir = training_dir+"/training_"+name+"/"+hyperpath
    
    graph = tf.Graph()
    with graph.as_default():
        # Load the data
#         dataset = tf.data.experimental.make_csv_dataset(
#             data_dir+'/'+dataname,
#             5000,
#             select_columns=['T',' p',' rho',' h'],
#             column_defaults=[tf.float64,tf.float64,tf.float64,tf.float64]
#         )
# Set up the graph from the inputs
#         stream = atu.make_datastream(dataset,batch_size=0,buffer_size=1000)
#         stream = tf.transpose(stream)
        data_all = np.load(data_dir+"water_lg_scaled_train.npy") # TODO
        dataset_all  = tf.data.Dataset.from_tensors(data_all).repeat()
        dataset_mini = tf.data.Dataset.from_tensor_slices(data_all).repeat()
        # test_data = np.load(data_dir+'/'+"water_lg_scaled_test.npy")
        # testset = tf.data.Dataset.from_tensors(test_data).repeat()
        
        stream_all = dataset_all.make_one_shot_iterator().get_next()
        stream_mini = dataset_mini.batch(1000).make_one_shot_iterator().get_next()
        global_step = tf.train.get_or_create_global_step()
        onum = tf.Variable(0,name="csv_output_num")
        ae = AutoencoderFactory(hyper,outerdim,innerdim,stream_mini, stream_all)
        ae._make_hess_train_step(stream_all)
        init = tf.global_variables_initializer()
        meta_graph_def = tf.train.export_meta_graph(filename=training_dir+"/final_graph.meta")
        
        # Add some more hooks
        loghook = tf.train.SummarySaverHook(
            summary_op=tf.summary.scalar("goal",ae.goal),
            save_steps=25,output_dir=training_dir)
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
        # Do hessian steps here and there
        @DoStuffHook(freq=n_epoch-1)
        def newthook(ctx,run_values):
            # Now do the Hessian step on the last layer
            for i in range(5):
                ctx.session.run(ae.newt_step)
        # set up the session
        session = tf.train.MonitoredTrainingSession(
            checkpoint_dir=training_dir,
            hooks=[loghook,stophook,saverhook,extrahook,newthook])
        # train away
        with session as sess:
            # Do the SGD rounds
            while not sess.should_stop():
                sess.run(ae.train_step)

    
