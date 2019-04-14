from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from visdom import Visdom
import numpy as np
import argparse

import plotting_routines as rout

DEFAULT_PORT = 8097
DEFAULT_HOSTNAME = "http://localhost"
parser = argparse.ArgumentParser(description='Demo arguments')
parser.add_argument('-port', metavar='port', type=int, default=DEFAULT_PORT,
                    help='port the visdom server is running on.')
parser.add_argument('-server', metavar='server', type=str,
                    default=DEFAULT_HOSTNAME,
                    help='Server address of the target to run the demo on.')
FLAGS = parser.parse_args()

viz = Visdom(port=FLAGS.port, server=FLAGS.server)

hub = "/Users/afq/Google Drive/networks/"
eoses = [
        "water_slgc_logp_64",
#         "water_lg",
#     "water_linear",
]

for eos in eoses:

    # netplots = rout.plot_networks(surfs)
    # Subplots
    simplots = rout.make_simulation_plot_list(hub+'test_databases/'+eos+'_testing.db',eos)
    for p in simplots:
        viz.plotlyplot(p, env=eos)
    # The monolithic plot
#     simplots = rout.plot_simulations(hub+'test_databases/'+eos+'_testing.db',eos)
#     viz.plotlyplot(simplots, win='win_'+eos, env=eos)
    surfs = rout.read_networks(hub+'training_'+eos)
    viz.update_window_opts('win_'+eos,{'width':500,'height':500})
    netplots = rout.generate_trisurf_plots(surfs)
    for n,p in netplots:
        viz.plotlyplot(p, win='net_'+eos+n, env=eos)
        viz.update_window_opts('net_'+eos+n,
                               {'width':200,'height':200})
    

    