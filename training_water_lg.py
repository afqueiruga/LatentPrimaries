if __name__=="__main__":
    training_dir = "/Users/afq/Google Drive/networks/"
    data_dir = "/Users/afq/Dropbox/ML/primaryautoencoder/data_files/"
#     data_dir = "/Users/afq/Documents/Dropbox/ML/primaryautoencoder/data_files/"
    dataset = "water_lg_sharded/*.csv"
    
    sets_to_try = [
#         {'type':'Classifying','args':[1,1, 6,12,'tanh']},
#         {'type':'Classifying','args':[1,1, 6,12,'sigmoid']},
#         {'type':'Classifying','args':[1,3, 6,12,'tanh'],'ini':'rand','cae':0},
#         {'type':'Classifying','args':[1,3, 6,12,'tanh'],'ini':'pT','cae':0},
#         {'type':'Classifying','args':[1,3, 6,12,'tanh'],'ini':'rhoh','cae':0},
#         {'type':'Classifying','args':[1,3, 6,12,'tanh'],'ini':'rand','cae':1.0},
#         {'type':'Classifying','args':[1,3, 6,12,'tanh'],'ini':'pT','cae':1.0},
#         {'type':'Classifying','args':[1,3, 6,12,'tanh'],'ini':'rhoh','cae':1.0},
        {'type':'Classifying','args':[1,3, 6,12,'sigmoid'],'ini':'rand','cae':0},
        {'type':'Classifying','args':[1,3, 6,12,'sigmoid'],'ini':'pT','cae':0},
        {'type':'Classifying','args':[1,3, 6,12,'sigmoid'],'ini':'rhoh','cae':0},
        {'type':'Classifying','args':[1,3, 6,12,'sigmoid'],'ini':'rand','cae':1.0},
        {'type':'Classifying','args':[1,3, 6,12,'sigmoid'],'ini':'pT','cae':1.0},
        {'type':'Classifying','args':[1,3, 6,12,'sigmoid'],'ini':'rhoh','cae':1.0},
#         {'type':'Classifying','args':[1,4, 6,12,'sigmoid']},
#         {'type':'Classifying','args':[1,5, 6,12,'sigmoid']},
#         {'type':'Classifying','args':[1,1, 6,12,'relu']},
#         {'type':'Deep','args':[1,[],1,[8,8,8]]},
#         {'type':'Deep','args':[1,[],2,[8,8,8]]},
#         {'type':'Deep','args':[1,[],3,[8,8,8]]},
#         {'type':'Poly','args':[1,5],'ini':'rand','cae':0},
#         {'type':'Poly','args':[1,5],'ini':'pT','cae':0},
#         {'type':'Poly','args':[1,5],'ini':'rhoh','cae':0},
#         {'type':'Poly','args':[1,6]},
#         {'type':'Poly','args':[1,7]},
    ]
    inner_penalty = [
        0.0,
        0.1,
    ]
#     n_epoch = 25000
    n_epoch = 1000
#     import joblib
#     @joblib.delayed
#     def job(S):
#         train_autoencoder("water_lg",dataset, 4,2,S,
#                           training_dir=training_dir,
#                           n_epoch = n_epoch)
#         return 1
#     joblib.Parallel(n_jobs=2)(job(S) for S in sets_to_try)
    import multiprocessing as multi
    import itertools
    def job(S):
        from autoencoder_trainer import train_autoencoder
        train_autoencoder("water_lg",dataset, 4,2,S,
                          training_dir=training_dir,
                          data_dir = data_dir,
                          n_epoch = n_epoch)
    p = multi.Pool(processes=4)
    p.map( job, sets_to_try )
#     for S in sets_to_try:
#         job(S)
