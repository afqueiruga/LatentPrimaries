if __name__=="__main__":
    training_dir = "/Users/afq/Google Drive/networks/"
    if True: #imac
        data_dir = "/Users/afq/Dropbox/ML/primaryautoencoder/data_files/"
    else:
        data_dir = "/Users/afq/Documents/Dropbox/ML/primaryautoencoder/data_files/"
    dataset = "water_iapw_logp_scaled"
#     caes = [ 0.0, 1.0 ]
    inis = [ 'rand','pT', 'rhoh' ]
    caes = [ 0.0, ]
    extend_train_params = lambda arch : \
        [ dict(ini=ini,cae=cae,**arch) for ini in inis for cae in caes ]
    archs_to_try = [
        {'type':'Classifying','args':[1,1, 6,12,'sigmoid']},
        {'type':'Classifying','args':[2,1, 6,12,'sigmoid']},
        {'type':'Classifying','args':[1,1, 12,24,'sigmoid']},
        {'type':'Classifying','args':[1,5, 6,12,'sigmoid']},
        {'type':'Classifying','args':[2,1, 12,24,'sigmoid']},
        {'type':'Classifying','args':[2,3, 12,24,'sigmoid']},
        {'type':'Classifying','args':[2,4, 12,24,'sigmoid']},
        {'type':'Classifying','args':[2,5, 12,24,'sigmoid']},
        {'type':'Classifying','args':[2,6, 18,36,'sigmoid']},
        {'type':'Classifying','args':[2,6, 24,48,'sigmoid']},
    ]
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
        train_autoencoder("water_lg",dataset, 4,2,S,
                          training_dir=training_dir,
                          data_dir = data_dir,
                          n_epoch = n_epoch,
                          image_freq = 20)
    if True: # Toggle parallel
        p = multi.Pool(processes=4)
        p.map( job, sets_to_try )
    else:
        for S in sets_to_try:
            job(S)
