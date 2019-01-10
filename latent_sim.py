import numpy as np
import tensorflow as tf
import afqstensorutils as atu

def unshift(x, scale):
    return (scale[2,:]-scale[1,:])*x + scale[0,:]
def shift(x,scale):
    return (x-scale[0,:])/(scale[2,:]-scale[1,:])

class LatentSim():
    """Object for storing a simulation state"""
    def __init__(self):
        "Constructor"
        self._graph = None
        self._sess = None
        self._vars = {}
        
    def load_model(self, location, scale_file="", logp=True):
        """Read the autoencoder from a tensorflow file"""
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)
#         with self._sess as sess:
        with self._graph.as_default():
            saver = tf.train.import_meta_graph(location+'/final_graph.meta')
        saver.restore(self._sess,location+'/final_variables')
        # hooks to the decoder
        self.i_q = self._graph.get_tensor_by_name('i_q:0') 
        self.o_x = self._graph.get_tensor_by_name('decode:0')
        # hooks to the encoder
        self.i_x = self._graph.get_tensor_by_name('i_x:0') 
        self.o_q = self._graph.get_tensor_by_name('encode:0')
        # The simulation inputs and outputs in the unscaled space
        # (the prior conditioning of the data put it on [-1,1])
        if scale_file:
            self.logp = logp
            self.scale = np.loadtxt(scale_file,skiprows=1,delimiter=',')
            unshifted = unshift(self.o_x,self.scale[:,0:4])
            if logp:
                self.o_s = tf.concat([ tf.expand_dims(unshifted[:,0], -1),
                                      tf.expand_dims(tf.math.exp(unshifted[:,1]), -1),
                                      unshifted[:,2:]], axis=1)
            else:
                shifted = self.i_s
                self.o_s = unshifted
        else:
            self.scale = None
            self.o_s = self.o_x
        
        self.o_dxdq = atu.vector_gradient(self.o_x, self.i_q)
        self.o_dsdq = atu.vector_gradient(self.o_s, self.i_q)
        
        self.dtype = self.i_x.dtype

    def decode(self, q):
        " Go from q to s"
        return self._sess.run(self.o_s, feed_dict={self.i_q:q})
    
    def encode(self, s):
        """From a q that satisfies s"""
        # The shifting is done outside of tensorflow
        if not self.scale is None:
            if self.logp:
                slog = s.copy()
                slog[:,1] = np.log(slog[:,1])
                s=slog
            x = shift(s, self.scale[:,0:4])
        else:
            x = s
        return self._sess.run(self.o_q, feed_dict={self.i_x:x})
    
    def find_point(self, T=None, p=None, rho=None, h=None):
        """Specify only two coordinates and find q!"""
        s0 = np.expand_dims(self.scale[0,:],axis=0).copy()
        if self.logp: s0[0,1] = np.exp(s0[0,1])
        idcs = []

        q0 = self.encode(s0)
        if not T is None:
            s0[0,0] = T
            idcs.append(0)
        if not p is None:
            s0[0,1] = p
            idcs.append(1)
        if not rho is None:
            s0[0,2] = rho
            idcs.append(2)
        if not h is None:
            s0[0,3] = h
            idcs.append(3)
        idcs = np.array(idcs, dtype=np.intc)
        for i in range(200):
            Rt,Kt = self._sess.run([self.o_s,self.o_dsdq],
                                feed_dict={self.i_q:q0})
            R = Rt[0,idcs]-s0[0,idcs]
            K = Kt[0,idcs,:]
            Dq = np.linalg.solve(K,-R)
#             if Dq.isnan():
#                 break
            nDq = np.linalg.norm(Dq)
            print nDq, q0
            if np.isnan(Dq).any(): break
#             print  min(1.0,nDq)*Dq/nDq
            q0[:] += min(0.5,nDq)*Dq/nDq
            if np.linalg.norm(Dq)<5.0e-7:
                break
        print "Found point at ", self.decode(q0), " after ",i," iterations."
        return q0
    
    def build_dae(self, method='BWEuler'):
        """Builds the differential algebraic equation and sets up the system
        lhs(q)=rhs
        K = d lhs / dq
        """
        aii = {'BWEuler':1.0,
                'Trap':0.5}[method]
        with self._graph.as_default():
            aii = self.regvar('aii',aii)
            Dt  = self.regvar('Dt',0.1)

            m,r = self.m_and_r( *tf.split(self.o_s,4,axis=-1 ))
            self.m = m
            self.r = r
            self.lhs = m - Dt*aii*r
            self.rhs = m + (1.0-aii)*Dt*r
            self.K_lhs = atu.vector_gradient(self.lhs,self.i_q)

            # Initialize parameters
            ini=tf.variables_initializer(self._vars.values())
        self._sess.run(ini)
        
    def regvar(self, name,val):
        """Register a variable into the system of equations."""
        self._vars[name]=tf.Variable(val,dtype=self.dtype)
        return self._vars[name]
    
    def m_and_r(self, T,p,rho,h):
        """The mass and rate equations. Latent sim will solve:
        d m/ dt = r
        """
        p_inf = self.regvar( "p_inf", 6.0e5)
        T_inf = self.regvar( "T_inf", 300.0)
        k_p =   self.regvar( "k_p", 0.0)
        k_T =   self.regvar( "k_T", 1000.0)
        mass_source = self.regvar("mass_source",0.0)
        heat_source = self.regvar("heat_source",0.0)
        
        m = tf.concat([rho,
                       rho*h-p],axis=-1)
        rate = tf.concat([
            k_p*(p_inf-p)+mass_source,
            k_T*(T_inf-T)+heat_source],axis=-1)
        return m, rate
    
    def set_params(self, **kwargs):
        """Set a value k = v in the current session."""
        for k,v in kwargs.items():
            self._vars[k].load(v,self._sess)
    
    def solve_a_time_step(self, q0):
        """Solve one timestep. Note that LatentSim is stateless in this regard."""
        qi = q0.copy()
        rhs_0 = self._sess.run(self.rhs,feed_dict={self.i_q:q0})
        for k in range(100):
            K_k,lhs_k = self._sess.run([self.K_lhs,self.lhs],feed_dict={self.i_q:qi})
            R = rhs_0 - lhs_k
            Dq = np.linalg.solve(K_k[0,:,:],R[0,:])
            nDq = np.linalg.norm(Dq)
            step = min(0.001/nDq,1.0)*Dq
            # TODO line search
            qi[:] += step
            if nDq<2.0e-7: break
        print k, nDq
        return qi
    
    def initialize(self, s):
        pass