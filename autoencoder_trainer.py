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
                     cae_lambda=hyper['cae'],)


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
                      training_dir='',data_dir='',n_epoch=5000, image_freq=1500,
                      UseNewt = True):
    hyperpath = string_identifier(hyper)
    training_dir = training_dir+"/training_"+name+"/"+hyperpath
    newt_freq = n_epoch
    graph = tf.Graph()
    with graph.as_default():
        # Load the data
        data_all = np.load(data_dir+"/"+dataname+"_train.npy")
        dataset_all  = tf.data.Dataset.from_tensors(data_all).repeat()
        dataset_mini = tf.data.Dataset.from_tensor_slices(data_all).repeat()
        test_data = np.load(data_dir+'/'+dataname+"_test.npy")
        dataset_test = tf.data.Dataset.from_tensors(test_data).repeat()
        # Make streams
        stream_all = dataset_all.make_one_shot_iterator().get_next()
        stream_mini = dataset_mini.batch(1000).make_one_shot_iterator().get_next()
        stream_test = dataset_test.make_one_shot_iterator().get_next()
        
        # Make the autoencoder graph
        global_step = tf.train.get_or_create_global_step()
        onum = tf.Variable(0,name="csv_output_num")
        ae = AutoencoderFactory(hyper,outerdim,innerdim,stream_mini, stream_all)
        if UseNewt: # Leave it out of the graph if so; it's heavy and expensive to compute
            ae._make_hess_train_step(stream_all)
        ae.goal_test = ae.make_goal(stream_test)
        init = tf.global_variables_initializer()
        meta_graph_def = tf.train.export_meta_graph(filename=training_dir+"/final_graph.meta")
        
        # Add some more hooks
        loghook = tf.train.SummarySaverHook(
            summary_op=[tf.summary.scalar("goaltrain",ae.goal_all),
                        tf.summary.scalar("goaltest",ae.goal_test)],
            save_steps=10,output_dir=training_dir)
        # Print to screen hook
        @DoStuffHook(freq=1)
        def printgoalhook(ctx,run_values):
            print(ctx.sessiormn.run(ae.goal_all))
        stophook = tf.train.StopAtStepHook(last_step=n_epoch)
        saverhook = SaveAtEndHook(training_dir+"/final_variables")
        # Make a closure into a hook
        @DoStuffHook(freq=image_freq)
        def extrahook(ctx,run_values):
            stepnum = global_step.eval(session=ctx.session)
            onum_val = onum.eval(session=ctx.session)
            onum_val+=1
            onum.load(onum_val, session=ctx.session)
            print("extra:",ae.goal.eval(session=ctx.session))
            header="T, p, rho, h"
            ae.save_fit(training_dir+"/surf_{0}.csv".format(onum_val),
                        header,sess=ctx.session)
        # Do hessian steps here and there
        def newtstep(sess):
            print("Running the newton step")
            for i in range(1):
                print("newt:",sess.run(ae.goal_all) )
                for v in ae._get_hess_vars():
                    print(sess.run(v))
                # x=ae.data.eval(session=sess)
                # W=ae.vars['dec_W_curve'].eval(session=sess)
                # b=ae.vars["dec_b_curve"].eval(session=sess)
                # q = ae.o_q.eval(session=sess,feed_dict={ae.i_x:x[0:1,:]})
                # p_hand = ae.o_class.eval(session=sess,feed_dict={ae.i_q:q})
                # f_hand = np.einsum('jik,Qj->Qik',W,q)+b
                # x_hand = np.einsum('Qik,Qi->Qk',f_hand,p_hand)
                # from IPython import embed ; embed()
                # sess.run(ae.newt_step)
                ae._do_hess_train_step(sess)
            print("newt:",sess.run([ae.goal_all]) )
            for v in ae._get_hess_vars():
                print(sess.run(v))
            sess.run(ae.sgd_reset)
        @DoStuffHook(freq=n_epoch/2)
        def newthook(ctx,run_values):
            # Now do the Hessian step on the last layer
            newtstep(ctx.session)

        # set up the session
        # replace with MonitoredSession with manually crafted
        # checkpointhook
        session = tf.train.MonitoredTrainingSession(
            checkpoint_dir=training_dir,
            hooks=[loghook,saverhook,extrahook,
                   #newthook,
                   stophook])
        # train away
        with session as sess:
            # Do the SGD rounds
            i=0
            while not sess.should_stop():
                # Do two newt steps throughout the training period
                if UseNewt and i%(newt_freq)==newt_freq-1:
                    newtstep(sess)
                else:
                    ae.update_beta(sess)
                    sess.run(ae.train_step)
#                     print("loop:",sess.run([ae.goal_all,ae.train_step]
#                                        +list(ae._get_hess_vars())) )
                    # increment beta
                    #try:
                i+=1


