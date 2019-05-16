if __name__=="__main__":
    training_dir = "/Users/afq/Google Drive/networks/"
    data_dir = "/Users/afq/Dropbox/ML/primaryautoencoder/data_files/"
#     data_dir = "/Users/afq/Documents/Dropbox/ML/primaryautoencoder/data_files/"
    dataset = "water_lg_sharded/*.csv"
#     inis = [ 'rand', 'pT', 'rhoh' ]
#     caes = [ 0.0, 1.0 ]
    inis = [ 'rand', ]# 'pT', 'rhoh' ]
    caes = [ 0.0 ]
    extend_train_params = lambda arch : \
        [ dict(ini=ini,cae=cae,**arch) for ini in inis for cae in caes ]
    archs_to_try = [
        {'type':'Classifying','args':[1,3, 3,3,'sigmoid']},
#         {'type':'Classifying','args':[1,3, 6,12,'sigmoid']},
#         {'type':'Classifying','args':[1,1, 6,12,'tanh']},
#         {'type':'Classifying','args':[1,1, 6,12,'sigmoid']},
#         {'type':'Poly','args':[1,4]},
#         {'type':'Poly','args':[1,5]},
    ]
    sets_to_try = []
    for arch in archs_to_try:
        sets_to_try.extend(extend_train_params(arch))

    n_epoch = 100

    import multiprocessing as multi
    import itertools
    def job(S):
        # Import of tensorflow has to happen *after* forking
        from autoencoder_trainer import train_autoencoder
        train_autoencoder("water_lg",dataset, 4,2,S,
                          training_dir=training_dir,
                          data_dir = data_dir,
                          n_epoch = n_epoch)
#     p = multi.Pool(processes=1)
#     p.map( job, sets_to_try )
    for S in sets_to_try:
        job(S)
