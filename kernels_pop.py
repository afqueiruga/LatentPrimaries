"""
These are a few kernels that LatentFlow will use to assemble tensorflow-based
kernels with cornflakes.
"""

import popcorn as pc

Vol_2Vec = pc.DofSpace(2,0,1)
Vol_4Vec = pc.DofSpace(4,0,1)
i_R = pc.Input('iR',Vol_2Vec)
i_K = pc.Input('iK',Vol_4Vec)
o_R = pc.Output('R',(Vol_2Vec,),1)
o_K = pc.Output('K',(Vol_2Vec,),2)
pc.Kernel('idty_R',listing=[
    pc.Asgn(o_R,i_R),
])
pc.Kernel('idty_RK',listing=[
    pc.Asgn(o_R,i_R),
    pc.Asgn(o_K,i_K.reshape(2,2)),

])


Vol_2Vec   = pc.DofSpace( 2,0,2)
Face_4Vec  = pc.DofSpace( 4,2,3)
Face_16Vec = pc.DofSpace(16,2,3)
i_R = pc.Input('iR',Face_4Vec)
i_K = pc.Input('iK',Face_16Vec)
o_R = pc.Output('R',(Vol_2Vec,),1)
o_K = pc.Output('K',(Vol_2Vec,),2)
pc.Kernel('idty_2_R',listing=[
    pc.Asgn(o_R,i_R),
])
pc.Kernel('idty_2_RK',listing=[
    pc.Asgn(o_R,i_R),
    pc.Asgn(o_K,i_K.reshape(4,4))
])

pc.Husk('identity')
