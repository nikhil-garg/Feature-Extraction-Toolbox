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
		dataset=["jc_mot","fp_im", "jc_im", "jm_im", "rr_im", "rh_im", "bp_im","wc_mot","zt_mot","fp_mot","gc_mot","hh_mot","hl_mot","jf_mot","jp_mot","rh_mot","rr_mot","ug_mot","jt_mot","jm_mot","gf_mot","bp_mot","cc_mot","ca_mot","de_mot"]
    )

    for args.dataset in parameters["dataset"]:	
        accd=NIKHIL_GIVE_NEW(args)
        df = df.append({"dataset":args.dataset,"tstep 500": accd['0'],"tstep 1000": accd['1'], "tstep 1500": accd['2'], "tstep 3000": accd['3']},ignore_index=True)

        log_file_name = 'accuracy_log_'+args.dataset+'.csv'
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