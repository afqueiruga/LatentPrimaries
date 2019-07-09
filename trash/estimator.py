from __future__ import print_function
from autoencoder import *

class MyHook(tf.train.SessionRunHook):
    def __init__(self,A, fname, save_freq=100):
        self.A = A
        self.save_freq = save_freq
        self.fname = fname
        self.onum = 0
        self.call_cnt = 0
    def after_run(self,run_context,run_values):
        self.call_cnt += 1
#         from IPython import embed ; embed()
        if self.call_cnt % self.save_freq == self.save_freq-1:
            self.A.save_fit(self.fname+"_"+str(self.onum)+".csv",
                            "T, p, rho, h",sess=run_context.session)
            self.onum += 1
            print("Saving ",self.onum,".")
    def end(self, run_context):
        saver = tf.train.Saver()
        saver.save(run_context,"./savemymodel")

# I need to generate an estimator for an autoencoder
# The autoencoder architecture should be an input
# I want to save that csv file to render in paraview
def my_input_fn():
    buffer_size = 1000
    batch_size=0
    dataset = tf.data.experimental.make_csv_dataset(
        'sharded/*.csv',
        1000,
        select_columns=['T',' p',' rho',' h']
    )
    def features_and_labels(x):
        return (tf.stack([x['T'],x[' p'],x[' rho'],x[' h']]),
                tf.stack([x['T'],x[' p'],x[' rho'],x[' h']]))
    dataset_stacked_and_copied = dataset.map(features_and_labels)
    nxt = dataset_stacked_and_copied.repeat()
    nxt = nxt.shuffle(buffer_size=buffer_size)
    if batch_size > 0:
        nxt = nxt.batch(batch_size)
    iterator = nxt.make_one_shot_iterator()
#     self.init_hook = lambda sess: sess.run(iterator.initialize())
#     from IPython import embed ; embed()
    features,labels = iterator.get_next()
    return tf.transpose(features),tf.transpose(labels)
    
def my_model_fn(features, labels, mode, params):
    # params is for me
    
    # make the model
    au = ClassifyingPolyAutoencoder(4,2, features, 1,2, 6,12)
    # Return an estimatorspec
    # For prediction:
    predictions = au.decode(au.encode(features))
    # For evaluation:
    loss = au.goal
    # For training:
    train_op = au.train_step
    meta_graph_def = tf.train.export_meta_graph(filename="savemymodel.meta")

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        export_outputs={'x',au.o_x},
        training_hooks=[MyHook(au,"auto_a/viz")])

# use tf.estimator.train_and_evaluate to batch experiments

class GlobHook(tf.train.SessionRunHook):
    def end(self, session_context):
        print("dawg")
        from IPython import embed ; embed()
def main(argv):
    # construct the estimator
    network = tf.estimator.Estimator(
        model_fn=my_model_fn,
        model_dir='auto_a',
        params={})
    # train it
    network.train(
        input_fn=my_input_fn,
        steps=100)
#     network.
#     hooks=[GlobHook()])
    # check its accuracy
#     score = network.evaluate(
#         input_fn=my_input_fn)
    # use it to make predictions
    # I'm going to optimize and save it instead
    
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)