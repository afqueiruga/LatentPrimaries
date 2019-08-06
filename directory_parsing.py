import glob, re
# Configure the eos hub. 
# This is factored out to not do imports before forking.
hubs = [
    "/Users/afq/Google Drive/networks/",
    "/Users/afq/Documents/Research/LBNL/eoshub/eoshub/networks/",
    "/Users/afq/Research/eoshub/networks/",   
]

test_dir = "/Users/afq/Research/eoshub/test_db/"

def get_list_of_eoses():
    eoses = []
    for hub in hubs:
        eos_dirs = glob.glob(hub+'/training_*')
        hub_eoses = [ k[(len(hub)+len('training_')):] 
                      for k in eos_dirs ]
        eoses.extend(hub_eoses)
    return eoses

def list_files(fpattern):
    files = glob.glob(fpattern)
    grab_digit = lambda f : int(re.search("([0-9]*)\.[a-zA-Z]*$",f).groups()[-1])
    files.sort(key=lambda f: grab_digit(f) )
    return files