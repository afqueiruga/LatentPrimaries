import numpy as np
import scipy.sparse.linalg
from matplotlib import pylab as plt

import cornflakes as cf
import husk_identity

from latent_sim import *

class LatentFlow(LatentSim):
    """Object for storing a simulation state, with support
    for assembling finite volume connections."""
    def __init__(self, model_loc, scale_file="", logp=True, method="BWEuler"):
        super(LatentFlow,self).__init__(model_loc,scale_file,logp,method)
        self.build_flux()
        # self.mesh = None
        
    def _make_dofmaps(self):
        Ny = self.X.shape[0]
        q = np.zeros(2*Ny)
        # Cornflakes Dofmaps describing how data is on the mesh
        self.dm_q = cf.Dofmap(2,0,2)
        self.dm_4 = cf.Dofmap(4,0,4)
        self.dm_face = cf.Dofmap_Strided(4,-4*Ny,4)
        self.dm_face16 = cf.Dofmap_Strided(16,-16*Ny,16)
        data = {'q':(q,self.dm_q),
                'X':(self.X,self.dm_q)} # Do I need this?
        
    def _make_vol_graph(self):
        """Cornflakes needs a graph of singletons to represent the volumes"""
        self.H_vol = cf.Hypergraph()
        for i in range(self.X.shape[0]):
            self.H_vol.Push_Edge([i])
        
    def make_line_mesh(self,Ny,L):
        """Makes a new mesh of elements in a line."""
        # The centers of the volumes
        self.X = np.c_[np.zeros((Ny,)),np.linspace(0,L,Ny)]
        # A graph of the connections
        self.H_face = cf.Hypergraph()
        for i in range(Ny-1):
            self.H_face.Push_Edge([i,i+1,i+Ny])
        self._make_vol_graph()
        self._make_dofmaps()
        
    def make_grid_mesh(self,Nx,Ny,W,H):
        """Makes a new mesh of elements in a grid."""
        self.X = cf.PP.init_grid(Nx,Ny,(0.5,0.5),(W-1.0,0),(0,H-1.0))
        self.H_face = cf.Graphers.Build_Pair_Graph(self.X, 1.1*W/Nx)
        self.H_face.Add_Edge_Vertex(self.X.shape[0])
        self._make_vol_graph()
        self._make_dofmaps()
        

    def plot(self,q, include_q=False, xlims=None):
        """Do a basic plot of the fields decoded by q.
        Assumes a line for now."""
        q = q.reshape(-1,2)
        s = self.decode(q)
        ncol = 4 if not include_q else 6
        fig = plt.figure()
        lines = []
        for i,leg in enumerate(['T','P','rho','rho*h']):
            plt.subplot(1,ncol,i+1)
            line = plt.plot(s[:,i], self.X[:,1],linewidth=5)[0]
            lines.append(line)
            plt.xlabel(leg)
            try:
                plt.xlim(*xlims[i])
            except TypeError as e:
                pass
        if include_q:
            for i,leg in enumerate(['q0','q1']):
                plt.subplot(1,ncol,i+4+1)
                line = plt.plot(q[:,i],self.X[:,1],linewidth=5)[0]
                lines.append(line)
                plt.xlabel(leg)
                try:
                    plt.xlim(*xlims[4+i])
                except TypeError as e:
                    pass
        return fig,lines
    
    def make_animation(self, qs, include_q = False, xlims=None):
        from matplotlib import animation
        
        if xlims == None:
            xlims = []
            templims = []
            for i in (0,-1):
                qlast = qs[-1].reshape(-1,2)
                s = self.decode(qlast)
                span = lambda x: (x.min() - 0.01*(x.max()-x.min()) - 0.01*x.min(),
                                    x.max() + 0.01*(x.max()-x.min())+ 0.01*x.max() )
                templims.append( [ span(x) for x in s.T ] + \
                        [ span(x) for x in qlast.T])
            for pairs in zip(*templims):
                xlims.append(( min([a for a,_ in pairs]), max([b for _,b in pairs])) )
        fig,lines = self.plot(qs[0],include_q,xlims)
        def animate(tidx):
            q = qs[tidx].reshape(-1,2)
            s = self.decode(q)
            for i,leg in enumerate(['T','P','rho','rho*h']):
                lines[i].set_data(s[:,i], self.X[:,1])
            if include_q:
                for i,leg in enumerate(['q0','q1']):
                    lines[4+i].set_data(q[:,i],self.X[:,1])
        return animation.FuncAnimation(fig,animate, frames=len(qs))
    
    
    def set_uniform(self,T=None, p=None, rho=None, rho_h=None):
        q = np.zeros(2*self.X.shape[0])
        q0 = self.find_point(T=T,p=p,rho=rho,rho_h=rho_h)
        for e in self.H_vol:
            q[self.dm_q.Get_List(e)] = q0
        return q
    
    def build_system(self,q,q0):
        """Build the matrix system of equations for the next Newton step."""
        DT = self._sess.run(self._vars['Dt'])
        R0_vol_arr = self._sess.run([self.rhs], feed_dict ={self.i_q:q0.reshape(-1,2)})
        # The mass component at t
        vol_R0, = cf.Assemble(husk_identity.kernel_idty_R,
                                self.H_vol,
                                {'iR':(R0_vol_arr,self.dm_q),},
                                {'R':(self.dm_q,),},
                                ndof=self.X.shape[0]*2)
        # Assemble the mass component at t+Dt:
        K_vol_arr,R_vol_arr = self._sess.run([self.K_lhs,self.lhs],
                                           feed_dict ={self.i_q:q.reshape(-1,2)})
        vol_R,vol_K = cf.Assemble(husk_identity.kernel_idty_RK,
                                    self.H_vol,
                                    {'iR':(R_vol_arr,self.dm_q),
                                     'iK':(K_vol_arr,self.dm_4)},
                                    {'R':(self.dm_q,),'K':(self.dm_q,)},
                                    ndof=self.X.shape[0]*2)
        # Assemble the fluxes:
        q_face = np.array([ q[self.dm_q.Get_List(e[0:2])] 
                           for e in self.H_face ])
        X_face = np.array([ self.X.ravel()[self.dm_q.Get_List(e[0:2])]
                            for e in self.H_face ])
        qA = q_face[:,0:2]
        qB = q_face[:,2:4]
        XA = X_face[:,0:2]
        XB = X_face[:,2:4]
        oF,oKF = self._sess.run([self.o_F,self.o_KF],feed_dict={
            self.i_q:qA,self.i_q2:qB,
            self.i_XA:XA,self.i_XB:XB})
        flux_R,flux_K=cf.Assemble(husk_identity.kernel_idty_2_RK,
                                    self.H_face,
                                    {'iR':(oF.flatten(),self.dm_face),
                                     'iK':(oKF.flatten(),self.dm_face16)},
                                    {'R':(self.dm_q,),'K':(self.dm_q,)},
                                    ndof=self.X.shape[0]*2)
        # Make the runge kutta system
        RR = vol_R0 - vol_R + DT*flux_R
        KK = vol_K - DT*flux_K
        return RR,KK
    
    def integrate_in_time(self, q0, N_steps):
        """Integrate for a number of time steps"""
        qs = []
        q0 = q0.copy()
        q = q0.copy()
        for t in range(N_steps):
            for itr in range(10):
                R,K = self.build_system(q,q0)
                Dq = scipy.sparse.linalg.spsolve(K,R)
                q[:] += Dq
                norm = np.linalg.norm(Dq)
                if norm < 1.0e-12:
                    break
            print("Converged in ",itr," steps.")
            q0[:] = q[:]
            qs.append(q.copy())
        return qs
    