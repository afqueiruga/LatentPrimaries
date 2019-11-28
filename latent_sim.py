from __future__ import print_function
import numpy as np
import tensorflow as tf
import afqstensorutils as atu

LOAD_PROTOBUF = True

def unshift(x, scale):
    return (scale[2,:]-scale[1,:])*x + scale[0,:]
def shift(x,scale):
    return (x-scale[0,:])/(scale[2,:]-scale[1,:])



class LatentSim():
    """Object for storing a simulation state"""
    def __init__(self, model_loc,scale_file="", logp=True, method="BWEuler"):
        "Constructor"
        self._graph = None
        self._sess = None
        self._vars = {}
        self.model_name = model_loc
        self.load_model(model_loc, scale_file,logp)
        self.build_dae(method)
        
    def load_model(self, location, scale_file="", logp=True):
        """Read the autoencoder from a tensorflow file"""
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)
        if LOAD_PROTOBUF:
            with tf.gfile.GFile(location+'/final_variables_frz','rb') as f:
                gdef = tf.GraphDef()
                gdef.ParseFromString(f.read())
            with self._graph.as_default():
                tf.import_graph_def(gdef)
            pfx='import/'
        else:
            with self._graph.as_default():
                saver = tf.train.import_meta_graph(location+'/final_variables.meta')
            saver.restore(self._sess,location+'/final_variables')
            pfx=''
        # hooks to the decoder
        self.i_q = self._graph.get_tensor_by_name(pfx+'i_q:0') 
        self.o_x = self._graph.get_tensor_by_name(pfx+'decode:0')
        # hooks to the encoder
        self.i_x = self._graph.get_tensor_by_name(pfx+'i_x:0')
        self.o_q = self._graph.get_tensor_by_name(pfx+'encode:0')
        # The simulation inputs and outputs in the unscaled space
        # (the prior conditioning of the data put it on [-1,1])
        if scale_file:
            self.logp = logp
            raw_scale = np.loadtxt(scale_file,skiprows=1,delimiter=',')
            self.scale = raw_scale[:,0:4]
            unshifted = unshift(self.o_x,self.scale[:,0:4])
            if logp:
                self.o_s = tf.concat([ tf.expand_dims(unshifted[:,0], -1),
                                      tf.expand_dims(tf.math.exp(unshifted[:,1]), -1),
                                      unshifted[:,2:]], axis=1)
            else:
                self.o_s = unshifted
        else:
            self.scale = None
            self.o_s = self.o_x
        
        self.o_dxdq = atu.vector_gradient_dep(self.o_x, self.i_q)
        self.o_dsdq = atu.vector_gradient_dep(self.o_s, self.i_q)
        
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
    
    def find_point(self, T=None, p=None, rho=None, rho_h=None, 
                   phase=None, under_relax=0.1,verbose=True):
        """Specify only two coordinates and find q!
        
        The phase tag only helps with the initial conditions.
        """
        s0_none = np.expand_dims(self.scale[0,:],axis=0).copy()
        if self.logp: s0_none[0,1] = np.exp(s0_none[0,1])
        guesses = {
            None: s0_none,
            "Gas": np.array([[450,1.0e2, 0.00048149, 0.00048149*2835269.40]]),
            "Liquid": np.array([[350,1.0e7,978.09, 978.09*329726.06]]),
#             "Solid": np.array([[250,1.0e5,919.87,-379930.33]]),
#             "Supercritical": np.array([[900,132167913.14,900,2964049.86]])
        }
        # One guess:
#         s0 = guesses[phase]
        # Try all of the guesses:
        ss = []
        for phase in guesses:
            s0 = guesses[phase]
            # Assign the initial condition and mark which we specified
            idcs = []
            if not T is None:
                s0[0,0] = T
                idcs.append(0)
            if not p is None:
                s0[0,1] = p
                idcs.append(1)
            if not rho is None:
                s0[0,2] = rho
                idcs.append(2)
            if not rho_h is None:
                s0[0,3] = rho_h
                idcs.append(3)
            if len(idcs)>2:
                raise RuntimeError("LatentSim: You specified too many variables.")
            idcs = np.array(idcs, dtype=np.intc)
            try:
                q,s = self._find_point(s0, idcs, 
                                    under_relax=under_relax,
                                    verbose=verbose)
                ss.append((q,s))
            except RuntimeError as e:
                pass
        if len(ss)==0:
            print("None of the initial guesses converged!")
            raise RuntimeError("None of the initial guesses converged!")
        dist_closest = 1.0e10
        s_closest = ss[0]
        q_closest = q[0]
        for q,s_found in ss:
            dist = np.linalg.norm([ 
                (s_found[0,c]-s0[0,c])/s0[0,c] 
                for c in idcs])
            if dist < dist_closest:
                s_closest = s_found
                q_closest = q
        return q_closest
    
    def _find_point(self, s0, idcs, under_relax=0.1,verbose=False):
        # Initial guess for q. TODO: Where should it be?
        q0 = self.encode(s0)
        # Iterate until the decoder is satisfied
        for i in range(100):
            Rt,Kt = self._sess.run([self.o_s,self.o_dsdq],
                                feed_dict={self.i_q:q0})
            # print(Rt,Kt)
            R = Rt[0,idcs]-s0[0,idcs]
            K = Kt[0,idcs,:]
            try:
                Dq = np.linalg.solve(K,-R)
            except Exception as e:
                print("Singular Matrix Error during initialization")
                raise RuntimeError("Initialization Failed")
            # print(R,K,Dq)
            nDq = np.linalg.norm(Dq)
#             print nDq, q0
            if np.isnan(Dq).any(): 
                print("Iteration failed with a nan.")
                raise RuntimeError("Initialization Failed")
            if np.linalg.norm(Dq)<1.0e-14:
                break # It might be 0
#             print  min(1.0,nDq)*Dq/nDq
            q0[:] += under_relax*Dq #min(0.5,nDq)*Dq/nDq
            if np.linalg.norm(Dq)<5.0e-7:
                break
        s_found = self.decode(q0)
        
#         if np.linalg.norm([ 
#             (s_found[0,c]-s0[0,c])/s0[0,c] 
#             for c in idcs]) > 0.0001:
#             if verbose: 
#                 print("Found point at ", s_found, " after ",i,
#                       " iterations, But that point was far away.")
#             raise RuntimeError("Couldn't find point")
        return q0, self.decode(q0)
    
    def build_dae(self, method='BWEuler'):
        """Builds the differential algebraic equation and sets up the system
        lhs(q)=rhs
        K = d lhs / dq
        """
        aii = {'BWEuler':1.0,
                'Trap':0.5}[method]
        with self._graph.as_default():
            aii = self.regvar('aii',aii)
            Dt  = self.regvar('Dt',1.0)

            T,p,rho,rho_h = tf.split(self.o_s,4,axis=-1)
            m,r = self.m_and_r( T,p,rho,rho_h )
            self.m = m
            self.r = r
            self.lhs = m - Dt*aii*r
            self.rhs = m + (1.0-aii)*Dt*r
            self.K_lhs = atu.vector_gradient_dep(self.lhs,self.i_q)

            # Initialize parameters
            ini=tf.variables_initializer(self._vars.values())
        self._sess.run(ini)
        
    def regvar(self, name,val):
        """Register a variable into the system of equations."""
        self._vars[name] = tf.Variable(val,dtype=self.dtype)
        return self._vars[name]
    
    def m_and_r(self, T,p,rho,rho_h):
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
                       rho_h-p],axis=-1)
        mflux = k_p*(p_inf-p)+mass_source
        rate = tf.concat([
            mflux,
            rho_h/rho*mflux + k_T*(T_inf-T)+heat_source],axis=-1)
        return m, rate
    
    def build_flux(self):
        """This should really go into the child LatentFlow."""
        with self._graph.as_default():
            self.i_XA = tf.placeholder(name='i_XA',shape=(None,2),dtype=self.dtype)
            self.i_XB = tf.placeholder(name='i_XB',shape=(None,2),dtype=self.dtype)
            self.i_q2 = tf.placeholder(name='i_q2',shape=(None,2),dtype=self.dtype)
            self.o_s2 = atu.replicate_subgraph(self.o_s,{self.i_q:self.i_q2})
            TA,pA,rhoA,rho_hA = tf.split(self.o_s,4,axis=-1)
            TB,pB,rhoB,rho_hB = tf.split(self.o_s2,4,axis=-1)
            self.o_F = self.flux(TA,pA,rhoA,rho_hA,
                          TB,pB,rhoB,rho_hB,
                          self.i_XA,self.i_XB)
            
            self.o_KF = tf.concat([atu.vector_gradient_dep(self.o_F,self.i_q),
                                   atu.vector_gradient_dep(self.o_F,self.i_q2)],axis=-1)
        self._sess.run(tf.variables_initializer(self._vars.values()))
        
    def flux(self,TA,pA,rhoA,rho_hA,
                  TB,pB,rhoB,rho_hB,
                  XA,XB):
        k_p =   self.regvar( "k_p", 10.0)
        k_T =   self.regvar( "k_T", 10.0)
        g = self.regvar("g",[0,-9.81])
        L = tf.norm(XA-XB,axis=-1)
        n = tf.einsum('ij,i->ij',(XB-XA),(1.0/L))
        rhobar = (rhoB + rhoA)/2.0
        rho_hbar = (rho_hB + rho_hA)/2.0
        grav_forcing = rhobar*tf.expand_dims(tf.einsum('j,ij->i',g, n),axis=-1 )
        mflux = k_p * ( tf.einsum('ij,i->ij',(pB - pA),(1.0/L) ) - grav_forcing )
        qflux = k_T * ( tf.einsum('ij,i->ij',(TB - TA),(1.0/L) ) )

        F = tf.concat([
            mflux,
            qflux + mflux*rho_hbar/rhobar,
            - mflux,
            - (qflux + mflux*rho_hbar/rhobar),
        ],axis=-1)
        return F
    
    def set_params(self, **kwargs):
        """Set a value k = v in the current session."""
        for k,v in kwargs.items():
            self._vars[k].load(v,self._sess)
    
    def solve_a_time_step(self, q0, under_relax=0.1, verbose=False):
        """Solve one timestep. Note that LatentSim is stateless in this 
        regard."""
        qi = q0.copy()
        rhs_0 = self._sess.run(self.rhs,feed_dict={self.i_q:q0})
        for k in range(100):
            K_k,lhs_k = self._sess.run([self.K_lhs,self.lhs], feed_dict ={self.i_q:qi})
            R = rhs_0 - lhs_k
            Dq = np.linalg.solve(K_k[0,:,:],R[0,:])
            nDq = np.linalg.norm(Dq)

            if nDq<1.0e-14: 
                break # It might be 0, we're done
            # step = min(0.001/nDq,1.0)*Dq
            step = under_relax*Dq
            # TODO line search
            if verbose:
                s = self.decode(qi)
                print(k, qi, s, nDq)
            qi[:] += step
            if nDq<2.0e-7: break
        return qi,k,nDq
    
    
    def integrate(self, t_max, q0,
                  schedule=lambda s,t :None,
                  verbose=False, under_relax=0.1):
        """Integrate from t=0 to t_max."""
        t = 0
        q = q0.copy()
        timeseries = [np.c_[t,q,self.decode(q)]]
        Dt = self._sess.run(self._vars['Dt'])
        while t<t_max:
            t+=Dt
            schedule(self, t)
            q,k,nDq = self.solve_a_time_step(q,under_relax = under_relax)
            if np.isnan(q).any() or np.isinf(q).any():
                print("NaN encountered at t=",t)
                break
            if nDq > 2.0e-7:
                print("Failed to converge at t=",t,
                      " |Dq| was ",nDq," after ",
                      k," iterations; quiting.")
                break
            if verbose:
                print(t,k,nDq)
            timeseries.append(np.c_[t,q,self.decode(q),k])
        return np.vstack(timeseries)
        