import numpy as np
import tensorflow as tf
import afqstensorutils as atu
from matplotlib import pylab as plt

def myev(x,feed_dict={},session=None):
    "Sanitized for interactive session contexts"
    if session:
        return session.run(x,feed_dict=feed_dict)
    else:
        return x.eval(feed_dict=feed_dict)
    
encoder_init_options = {
    "pT":np.array([[1,0],[0,1],[0,0],[0,0]],dtype=np.float64),
    "rhoh":np.array([[0,0],[0,0],[1,0],[0,1]],dtype=np.float64),
    "rand":np.array([[0,0],[0,0],[0,0],[0,0]],dtype=np.float64),
}
    
class Autoencoder(object):
    """
    A class for generating autoencoders.

    This parentclass is a linear map that's (maybe?) principal component analysis.
    """
    def __init__(self, size_x, size_q, data, data_all=None,
                 encoder_init="rand",
                 cae_lambda=0):
        self.size_x = size_x
        self.size_q = size_q
        self.dtype = data.dtype
        # initialization options
        self.encoder_init = encoder_init
        self.cae_lambda = cae_lambda
        # Storage for my variables
        self.vars = {}
        # Make the trainer
        self.data = data
        self.goal = self.make_goal(data)
        self.train_step = self._make_train_step(data)
        if data_all is None: data_all = data
        self.goal_all = self.make_goal(data_all)
        self.newt_step = None

        # Make evaluating graphs
        self.i_x = tf.placeholder(name='i_x', shape=(None,size_x,), dtype=self.dtype )
        self.o_q = self.encode( self.i_x, name='encode' )
        self.i_q = tf.placeholder(name='i_q', shape=(None,size_q,), dtype=self.dtype )
        self.o_x = self.decode( self.i_q, name='decode' )
#         self.o_grad_x = atu.vector_gradient(self.o_x, self.i_q)
        # Make the loggers for tensorboard
        
    def encode(self, x, name=None):
        W = self._var( "enc_W", (self.size_q, self.size_x),
                     initial_value=encoder_init_options[self.encoder_init])
        b = self._var( "enc_b", (self.size_q,) )
        return tf.add(tf.matmul(W,x),b)
    
    def decode(self, q, name=None):
        W = self._var( "dec_W", (self.size_x, self.size_q) )
        b = self._var( "dec_b", (self.size_x,) )
        x = tf.matmul(W,q)+b
        return tf.identity(x,)
    
    def make_goal(self, data):
        q = self.encode(data)
        pred = self.decode(q)
        loss = tf.losses.mean_squared_error(data, pred)
        if self.cae_lambda != 0:
            dqdx = atu.vector_gradient_dep(q,data)
            cae = tf.constant(self.cae_lambda,dtype=self.dtype) \
                    * tf.norm(dqdx, axis=(1,2))
            return loss + tf.metrics.mean(cae)[0] # CHECK
        else:
            return loss

    def _make_train_step(self, data):
        opt = tf.train.AdamOptimizer(1e-2)
        #opt = tf.train.GradientDescentOptimizer(1e-3)
        #opt = tf.train.RMSPropOptimizer(1e-2)
        loss = self.make_goal(data)
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        ts = opt.minimize(loss,global_step=tf.train.get_or_create_global_step(),var_list=var_list)
        self.sgd_opt = opt
        self.sgd_step = ts
        opt_vars = [opt.get_slot(var, name) for name in opt.get_slot_names() for var in var_list]
        try:
            opt_vars += opt._get_beta_accumulators() # only for adam
        except AttributeError:
            print("Wasn't ADAM")
            pass
        opt_vars = [ v for v in opt_vars if isinstance(v, tf.Variable) ]
        print(opt_vars)
        self.sgd_reset = tf.variables_initializer(opt_vars)
        return ts
    
    def _get_hess_vars(self):
        """These are variables used for the final fitting
        phase."""
        return (self.vars["dec_W"], self.vars["dec_b"])
        
    def _make_hess_train_step(self,data):
        loss = self.make_goal(data)
        ops = atu.NewtonsMethod_pieces(loss, self._get_hess_vars())
        self.newt_ops = ops
        self.newt_step = ops[0]
        return self.newt_step
    
    def _do_hess_train_step(self,session):
        G = self.newt_ops[1].eval(session=session)
        H = self.newt_ops[2].eval(session=session)
        idxs=np.where(~H.any(axis=1))[0]
        for i in idxs: 
            H[i,i]=1
        DeltaW = np.linalg.solve(H,-G)
        i_delta_W = self.newt_ops[-2]
        delta_assign_op = self.newt_ops[-1]
        session.run(delta_assign_op,feed_dict={i_delta_W:DeltaW.reshape(-1,1)})
        
    def eval_q(self, i_x):
        return sess.eval( self.o_q, feed_dict={self.i_x:i_x} )

    def _var(self, name, shape, stddev=0.1, 
             initial_value=None):
        try:
            v = self.vars[name]
            print("Fetching ",name)
        except KeyError:
            print("Making ",name)
            if not initial_value is None:
                ini = tf.constant(initial_value,dtype=self.dtype) + tf.truncated_normal(shape=shape,
                       stddev=stddev, dtype=self.dtype)
            else:
                ini = tf.truncated_normal(shape=shape,
                       stddev=stddev, dtype=self.dtype)
            v = tf.Variable(
                  ini,
                  name=name,
                  dtype=self.dtype)
            self.vars[name] = v
        return v
    
    def plot_distance(self, idxs=[0,1]):
        from matplotlib import pylab as plt
        XI,YI = idxs
        inputs = self.data.eval()
        q_enc = self.o_q.eval(feed_dict={self.i_x:inputs})
        x_dec = self.o_x.eval(feed_dict={self.i_q:q_enc})
        plt.plot( inputs[:,XI], inputs[:,YI],',')
        for a,b in zip(inputs,x_dec)[::10]:
            plt.plot( [a[XI],b[XI]], [a[YI],b[YI]],'-k')
        plt.plot(x_dec[:,XI],x_dec[:,YI],'+')
        plt.axis('square');
        
    def _extra_saves(self, ixs,qs,session=None):
        return "",[]
    
    def save_fit(self, fname, header,sess=None,samples=1):
        qs = []
        ixs = []
        
        for j in range(samples):
            ix = myev(self.data,session=sess)
            qs.append(myev(self.o_q,feed_dict={self.i_x:ix},session=sess))
            ixs.append(ix)
        qs = np.vstack(qs)
        ixs = np.vstack(ixs)
        oxs = myev(self.o_x,{self.i_q:qs},session=sess)
        
        extra_header,extra_saves = self._extra_saves(ixs,qs,session=sess)
        
        errors = ((ixs-oxs)**2).sum(axis=-1)
        
        from matplotlib import pylab as plt
        plt.plot(oxs[:,0],oxs[:,1],',')
        if extra_header:
            dat = np.hstack([oxs,qs, errors.reshape(-1,1),extra_saves.reshape((-1,1))])
        else:
            dat = np.hstack([oxs,qs, errors.reshape(-1,1)])
        auto_header = ","+",".join(["q{0}".format(i) for i in range(self.size_q)]) + ",error"
        np.savetxt(fname,dat,delimiter=",",
                   header=header+auto_header+extra_header,comments="")
        
        
class PolyAutoencoder(Autoencoder):
    """
    The basic implementation. Reduces to the linear case when the 
    polynomial bases are set to 0 on either side.
    """
    def __init__(self, size_x, size_q, data, Np_enc, Np_dec, 
                 data_all=None, encoder_init="rand",
                 cae_lambda=0):
        self.Np_enc = Np_enc
        self.Np_dec = Np_dec
        Autoencoder.__init__(self,size_x, size_q, data, data_all,
                            encoder_init=encoder_init,
                            cae_lambda=cae_lambda)
        
    def encode(self, x, name=None):
        N_coeff = atu.Npolyexpand( self.size_x, self.Np_enc )
        We1 = self._var("enc_W", (N_coeff, self.size_q),
                        initial_value = encoder_init_options[self.encoder_init])
        be1 = self._var("enc_b", (self.size_q,) )
        q = tf.matmul( atu.polyexpand(x, self.Np_enc), We1 ) + be1
        return tf.identity(q,name=name)
    
    def decode(self, q, name=None):
        N_coeff = atu.Npolyexpand( self.size_q, self.Np_dec )
        We1 = self._var("dec_W", (N_coeff, self.size_x) )
        be1 = self._var("dec_b", (self.size_x,) )
        x = tf.matmul( atu.polyexpand(q, self.Np_dec), We1 ) + be1
        return tf.identity(x,name=name)
    
    
class DeepPolyAutoencoder(Autoencoder):
    """
    The Deep relu'ed layers on each side
    """
    def __init__(self, size_x, size_q, data, Np_enc, enc_layers, 
                 Np_dec, dec_layers,
                 data_all=None, encoder_init="rand", cae_lambda=0):
        self.Np_enc = Np_enc
        self.Np_dec = Np_dec
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        Autoencoder.__init__(self,size_x, size_q, data, data_all,
                            encoder_init=encoder_init,
                            cae_lambda=cae_lambda)
        
    def _layers(self, inp, sizes, prefix=""):
        nxt = inp
        for i,hidden_size in enumerate(sizes):
            We1 = self._var(prefix+"_W"+str(i), (int(nxt.shape[-1]), hidden_size) )
            be1 = self._var(prefix+"_b"+str(i), (hidden_size,) )
            nxt = tf.nn.leaky_relu(tf.matmul( nxt, We1 ) + be1)
        return nxt
    
    def encode(self, x, name=None):
        N_coeff = atu.Npolyexpand( self.size_x, self.Np_enc )
        nxt = atu.polyexpand(x, self.Np_enc)
        nxt = self._layers(nxt, self.enc_layers + [self.size_q], prefix="enc")
        return tf.identity(nxt)
    
    def decode(self, q, name=None):
        N_coeff = atu.Npolyexpand( self.size_q, self.Np_dec )
        nxt = atu.polyexpand(q, self.Np_dec)
        nxt = self._layers(nxt, self.dec_layers + [self.size_x], prefix="dec")
        return tf.identity(nxt)
    
    
class ClassifyingPolyAutoencoder(Autoencoder):
    """
    This one is like the PolyAutoencoder, but has a classifying branch

    The idea is that this one learns piecewise branching logic.
    
    x > E > z -> Npoly ->   W1        -> * -> x
                  `-> W2->relu->softmax -^ (phase)
    
    """
    activations={
        'tanh':tf.tanh,
        'sigmoid':tf.sigmoid,
        'relu':tf.nn.relu
    }
    def __init__(self, size_x, size_q, data, p_enc, 
                 p_dec, N_curve, N_bound,
                 boundary_activation='tanh', softmax_it=True,
                 data_all=None,
                 encoder_init="rand",
                 cae_lambda=0):
        self.Np_enc = p_enc
        self.Np_dec = p_dec

        self.N_curve = N_curve
        self.N_bound = N_bound
        self.boundary_activation = boundary_activation
        self.softmax_it = softmax_it
        self.softmax_beta = tf.Variable(1.0, dtype=data.dtype,
                                        trainable=False)
        Autoencoder.__init__(self,size_x, size_q, data, data_all,
                            encoder_init=encoder_init,
                            cae_lambda=cae_lambda)
        self.o_class = self.classify(self.i_q)
        self.o_x_argmax = self.decode(self.i_q,phase_act="argmax")
        
    def encode(self, x, name=None):
        N_coeff = atu.Npolyexpand( self.size_x, self.Np_enc )
        W = self._var("enc_W", (N_coeff, self.size_q), 
                     initial_value=encoder_init_options[self.encoder_init])
        b = self._var("enc_b", (self.size_q,) )
        return tf.add(tf.matmul( atu.polyexpand(x, self.Np_enc), W ), b)
    
    def classify(self, q, name=None, phase_act="softmax"):
        qpoly = atu.polyexpand(q, self.Np_dec)
        N_coeff = atu.Npolyexpand( self.size_q, self.Np_dec )

        W2 = self._var("dec_W_bound", (N_coeff, self.N_bound) )
        b2 = self._var("dec_b_bound", (self.N_bound,) )
        act = self.activations[self.boundary_activation]
        h_bound = act(tf.tensordot(qpoly,W2,axes=[-1,0])+b2)
        
        W_select = self._var("dec_W_select", (self.N_bound,self.N_curve))
        h_select = tf.tensordot(h_bound,W_select, axes=[-1,0])
        if phase_act=="softmax":
            h_select = tf.nn.softmax(self.softmax_beta*h_select
                                     ,name=name)
        else:
            h_select = tf.one_hot(
                tf.argmax(h_select,axis=-1),
                depth=h_select.shape[-1],
                dtype=h_select.dtype,name=name)
            print("Did this.")
        return h_select
    
    def decode(self, q, name=None, phase_act="softmax"):
        qpoly = atu.polyexpand(q, self.Np_dec)
        N_coeff = atu.Npolyexpand( self.size_q, self.Np_dec )
        
        W1 = self._var("dec_W_curve", (N_coeff, self.N_curve, self.size_x) )
        b1 = self._var("dec_b_curve", (self.N_curve, self.size_x) )
        h_curve = tf.tensordot(qpoly,W1,axes=[-1,0])+b1
        
        h_select = self.classify(q,phase_act=phase_act)
        
        x = tf.einsum('ijk,ij->ik',h_curve,h_select)
        return tf.identity(x,name=name)
    
    def _extra_saves(self, ixs,qs, session=None):
        probs = myev(self.o_class,feed_dict={self.i_q:qs},session=session)
        classes = probs.argmax(axis=-1)
        return ",classes",classes

    def make_goal(self, data, phase_act="softmax"):
        q = self.encode(data)
        pred = self.decode(q,phase_act=phase_act)
        loss = tf.losses.mean_squared_error(data, pred)
        if self.cae_lambda != 0:
            dqdx = atu.vector_gradient_dep(q,data)
            cae = tf.constant(self.cae_lambda,dtype=self.dtype) \
                    * tf.norm(dqdx, axis=(1,2))
            return loss + tf.metrics.mean(cae)[0] # CHECK
        else:
            return loss
        
    def _get_hess_vars(self):
        """These are variables used for the final fitting
        phase."""
        return (self.vars["dec_W_curve"],)# self.vars["dec_b_curve"])
    
    def _make_hess_train_step(self,data):
        loss = self.make_goal(data,phase_act="onemax")
        ops = atu.NewtonsMethod_pieces(loss, self._get_hess_vars())
        self.newt_ops = ops
        self.newt_step = ops[0]
        return self.newt_step
