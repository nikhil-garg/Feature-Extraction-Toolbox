import itertools
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import os

#from evaluate_reservoir import *
#from utilis import *
from args_GIVE_NKHIL import args as my_args
from NIKHIL_GIVE_NEW import  *
from itertools import product
import time

if __name__ == '__main__':

    args = my_args()
    print(args.__dict__)
	# Fix the seed of all random number generator
    seed = 50
    random.seed(seed)
    np.random.seed(seed)
    df = pd.DataFrame({"dataset":[],"tstep 500": [],"tstep 1000": [], "tstep 1500": [], "tstep 3000": []})

    parameters = dict(
		dataset=["bci3","bp_im","bp_mot","ca_mot", "cc_mot","de_mot","fp_im","fp_mot","gc_mot","gf_mot","hh_mot","hl_mot","jc_im","jc_mot","jf_mot","jm_im","jm_mot","jp_mot","jt_mot","rh_im","rh_mot","rr_im","ug_mot","wc_mot","zt_mot"]
    )
    dataset_list = parameters["dataset"]
    # for args.dataset in parameters["dataset"]:	
    args.dataset = dataset_list[args.run]
    print('dataset:',args.dataset)
    accd=NIKHIL_GIVE_NEW(args)
    df = df.append({"dataset":args.dataset,"tstep 500": accd['0'],"tstep 1000": accd['1'], "tstep 1500": accd['2'], "tstep 3000": accd['3']},ignore_index=True)

    log_file_name = 'accuracy_log_'+str(args.run)+'.csv'
    pwd = os.getcwd()
    log_dir = pwd+'/log_dir/'
    df.to_csv(log_dir+log_file_name, index=False)

    df.to_csv(log_file_name, index=False)

    accuracy_df = pd.DataFrame({"Tstep": [500,1000,1500,3000], "Accuracy":[accd['0'], accd['1'], accd['2'], accd['3']]})
    # plot the feature importances in bars.
    plt.figure(figsize=(40,10))
    #plt.xticks(rotation=45)
    sns.set(font_scale=2)
    sns.lineplot(x="Tstep",y= "Accuracy", data=accuracy_df)
    plt.savefig(pwd+'/figures/'+args.dataset+'_accuracy.png')
    plt.tight_layout()
    plt.show()
        # logger.info('All done.')