if __name__=="__main__":
    training_dir = "/Users/afq/Research/eoshub/networks/"
    if True: #imac
        data_dir = "/Users/afq/Dropbox/ML/primaryautoencoder/data_files/"
    else:
        data_dir = "/Users/afq/Documents/Dropbox/ML/primaryautoencoder/data_files/"
    dataset = "water_iapws_lg_scaled"
    eosname = "water_iapws_rh_lg"
    inis = [ 'rand','pT', 'rhoh' ]
    caes = [ 0.0, 1.0 ]
    extend_train_params = lambda arch : \
        [ dict(ini=ini,cae=cae,**arch) for ini in inis for cae in caes ]
    
    class_archs = [
        [1,1, 3,2,'sigmoid'],
        [1,1, 3,3,'sigmoid'],
        [1,1, 6,12,'sigmoid'],
        [1,2, 3,2,'sigmoid'],
        [1,2, 3,3,'sigmoid'],
        [1,2, 6,12,'sigmoid'],
        [1,3, 3,3,'sigmoid'],
        [1,3, 3,4,'sigmoid'],
        [1,3, 3,5,'sigmoid'],
        [1,3, 6,12,'sigmoid'],
    ]
    
    beta_incs = [0.0,0.05]
    archs_to_try = []
    for arch in class_archs:
        for beta_inc in beta_incs:
            archs_to_try.append(
                {'type':'Classifying','args':arch+[beta_inc]}
            )
    
    sets_to_try = []
    for arch in archs_to_try:
        sets_to_try.extend(extend_train_params(arch))

    n_epoch = 5000

    import multiprocessing as multi
    import itertools
    def job(S):
        # Import of tensorflow has to happen *after* forking
        print(S)
        from autoencoder_trainer import train_autoencoder
        train_autoencoder(eosname ,dataset, 4,2,S,
                          training_dir=training_dir,
                          data_dir = data_dir,
                          n_epoch = n_epoch,
                          image_freq = 250)
    if True: # Toggle parallel
        p = multi.Pool(processes=4)
        p.map( job, sets_to_try )
    else:
        for S in sets_to_try:
            job(S)
