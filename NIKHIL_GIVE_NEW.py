import numpy as np
import pandas as pd
import time
import statsmodels.api as sm
from scipy import signal
from sklearn.preprocessing import StandardScaler
import EEGExtract as eeg
import pywt
from scipy.signal import hilbert, chirp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 
# Import the RFE from sklearn library
from sklearn.feature_selection import RFE,SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn import metrics
from genetic_selection import GeneticSelectionCV
import seaborn as sns
import os
from args_GIVE_NKHIL import args as my_args

args=my_args()

#data_temp1=np.array([])
def segment(data_trial, segment_length=500):
  data_final=np.array([])
  for i in range(0, data_trial.shape[0]):
    data_temp=data_trial[i,:,:]
    data_temp2=np.array([])
    for j in range(int(data_temp.shape[1]/segment_length)):
      llim=j*segment_length
      data_temp1=data_temp[:,llim:llim+segment_length]
      if j==0:
        data_temp2=data_temp1[np.newaxis,:,:]
      else:
        data_temp2=np.vstack((data_temp2, data_temp1[np.newaxis,:,:]))

    if i==0:
      data_final=data_temp2

    else:
      data_final=np.vstack((data_final, data_temp2))

  return data_final

def createFV_individual(data_train, data_test, fs):

  #subsampling by 4 
  
  data_2_subs=data_train
  '''data_2_subs=np.zeros((data_train.shape[0], data_train.shape[1], int(data_train.shape[2]/4)))
  for i in range(0, data_train.shape[0]):
      for j in range(0, data_train.shape[1]):
          data_2_subs[i, j, :]=signal.resample(data_2_sub[i, j, :], int(data_train.shape[2]/4))'''

  #data_2_subs.shape

  #Common Average Reference
  for j in range(0, data_train.shape[0]):
      car=np.zeros((data_2_subs.shape[2],))
      for i in range(0, data_train.shape[1]):
          car= car + data_2_subs[j,i,:]

      car=car/data_train.shape[1]
      #car.shape

      for k in range(0, data_train.shape[1]):
          data_2_subs[j,k,:]=data_2_subs[j,k,:]-car

  #Standard Scaler

  for j in range(0, data_train.shape[0]):
      kr=data_2_subs[j,:,:]
      kr=data_2_subs[j,:,:]
      
      scaler=StandardScaler().fit(kr.T)
      data_2_subs[j,:,:]=scaler.transform(kr.T).T

  '''#bandpass filter
  b, a = signal.butter(2, 0.4, 'low', analog=False)
  data_2_subs = signal.filtfilt(b, a, data_2_subs, axis=2)'''

  #Extracting all the features and concatenating them 
  final = np.array([])
  for j in range(0, data_2_subs.shape[0]):
      data_trial=data_2_subs[j,:,:].T
      #data_trial.shape

      data_trial_s1=data_trial[0:int(data_2_subs.shape[2]/3),:]
      #print(data_trial_s1.shape)
      data_trial_s2=data_trial[int(data_2_subs.shape[2]/3):2*int(data_2_subs.shape[2]/3),:]
      #print(data_trial_s2.shape)
      data_trial_s3=data_trial[2*int(data_2_subs.shape[2]/3):3*int(data_2_subs.shape[2]/3),:]
      #print(data_trial_s3.shape)

      #AR Coefficients

      #from statsmodels.datasets.sunspots import load
      #data = load()
      ARFV=np.array([])

      for i in range(0, data_train.shape[1]):
          rho1, sigma1 = sm.regression.linear_model.burg(data_trial_s1[:,i], order=2)
          rho2, sigma2 = sm.regression.linear_model.burg(data_trial_s2[:,i], order=2)
          rho3, sigma3 = sm.regression.linear_model.burg(data_trial_s3[:,i], order=2)
          ARFV=np.append(ARFV, (rho1, rho2, rho3))

      #print(ARFV) 

      #Haar wavelet

      HWDFV=np.array([])
      for i in range(0, data_train.shape[1]):
          (cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          HWDFV=np.append(HWDFV, cA)

      #Spectral Power estimates
      SPFV=np.array([])
      for i in range(0, data_train.shape[1]):
          f1, Pxx_den1 = signal.welch(data_trial_s1[:,i], int(data_2_subs.shape[2]/3))
          f2, Pxx_den2 = signal.welch(data_trial_s2[:,i], int(data_2_subs.shape[2]/3))
          f3, Pxx_den3 = signal.welch(data_trial_s3[:,i], int(data_2_subs.shape[2]/3))
          SPFV=np.append(SPFV, (Pxx_den1, Pxx_den2, Pxx_den3))

      #Concatenaton of All the feature vectors
      concated=np.concatenate((ARFV, HWDFV, SPFV), axis=None)
      concated=np.reshape(concated, (-1, 1))
      if j==0:
          final=concated
      else:
          final= np.hstack((final, concated))
      # print(j)
  print(final.shape)

  final=final.T

  eegData=np.rollaxis(data_2_subs, 0, 3)
  eegData.shape



  # Subband Information Quantity
  # delta (0.5–4 Hz)
  eegData_delta = eeg.filt_data(eegData, 0.5, 4, fs)
  ShannonRes_delta = eeg.shannonEntropy(eegData_delta, bin_min=-200, bin_max=200, binWidth=2)
  # theta (4–8 Hz)
  eegData_theta = eeg.filt_data(eegData, 4, 8, fs)
  ShannonRes_theta = eeg.shannonEntropy(eegData_theta, bin_min=-200, bin_max=200, binWidth=2)
  # alpha (8–12 Hz)
  eegData_alpha = eeg.filt_data(eegData, 8, 12, fs)
  ShannonRes_alpha = eeg.shannonEntropy(eegData_alpha, bin_min=-200, bin_max=200, binWidth=2)
  # beta (12–30 Hz)
  eegData_beta = eeg.filt_data(eegData, 12, 30, fs)
  ShannonRes_beta = eeg.shannonEntropy(eegData_beta, bin_min=-200, bin_max=200, binWidth=2)
  # gamma (30–100 Hz)
  eegData_gamma = eeg.filt_data(eegData, 30, 80, fs)
  ShannonRes_gamma = eeg.shannonEntropy(eegData_gamma, bin_min=-200, bin_max=200, binWidth=2)

  # Hjorth Mobility
  # Hjorth Complexity
  HjorthMob, HjorthComp = eeg.hjorthParameters(eegData)


  # Median Frequency
  medianFreqRes = eeg.medianFreq(eegData,fs)

  # Standard Deviation
  std_res = eeg.eegStd(eegData)

  # Regularity (burst-suppression)
  regularity_res = eeg.eegRegularity(eegData,fs)

  # Spikes
  minNumSamples = int(70*fs/1000)
  spikeNum_res = eeg.spikeNum(eegData,minNumSamples)

  # Sharp spike
  sharpSpike_res = eeg.shortSpikeNum(eegData,minNumSamples)
  
  #remove from here REMPOMOMOPMPOM
  concated=np.concatenate((ShannonRes_delta.T, ShannonRes_theta.T, ShannonRes_alpha.T, ShannonRes_beta.T, ShannonRes_gamma.T, HjorthMob.T, HjorthComp.T, medianFreqRes.T, std_res.T, regularity_res.T, spikeNum_res.T, sharpSpike_res.T), axis=1)

  final=np.hstack((final, concated))

  # δ band Power
  bandPwr_delta = eeg.bandPower(eegData, 0.5, 4, fs)
  # θ band Power
  bandPwr_theta = eeg.bandPower(eegData, 4, 8, fs)
  # α band Power
  bandPwr_alpha = eeg.bandPower(eegData, 8, 12, fs)
  # β band Power
  bandPwr_beta = eeg.bandPower(eegData, 12, 30, fs)
  # γ band Power
  bandPwr_gamma = eeg.bandPower(eegData, 30, 80, fs)

  concated_n=bandPwr_gamma.T
  final=np.hstack((final, concated_n))

  

  HTFV=np.array([])
  for j in range(0, eegData.shape[2]):
    eegData_temp=eegData[:,:,j]
    HTFV_temp=np.array([])
    for i in range(0, eegData.shape[0]):
      HTFV_temp=np.append(HTFV_temp, np.imag(hilbert(eegData_temp[i,:])))
    if(j==0):
      HTFV=HTFV_temp
    else:
      HTFV=np.vstack((HTFV, HTFV_temp))
    # print(j)

  final=np.hstack((final, HTFV))
  final.shape



  #subsampling by 4 
  data_2_subs_t=data_test
  '''data_2_subs_t=np.zeros((data_test.shape[0], data_test.shape[1], int(data_test.shape[2]/4)))
  for i in range(0, data_test.shape[0]):
      for j in range(0, data_test.shape[1]):
          data_2_subs_t[i, j, :]=signal.resample(data_2_sub_t[i, j, :], int(data_test.shape[2]/4))'''

  #data_2_subs_t.shape
  #Common Average Reference
  for j in range(0, data_2_subs_t.shape[0]):
      car=np.zeros((data_2_subs_t.shape[2],))
      for i in range(0, data_2_subs_t.shape[1]):
          car= car + data_2_subs_t[j,i,:]

      car=car/data_2_subs_t.shape[1]
      #car.shape

      for k in range(0, data_2_subs_t.shape[1]):
          data_2_subs_t[j,k,:]=data_2_subs_t[j,k,:]-car

  #Standard Scaler

  for j in range(0, data_2_subs_t.shape[0]):
      kr=data_2_subs_t[j,:,:]
      
      scaler=StandardScaler().fit(kr.T)
      data_2_subs_t[j,:,:]=scaler.transform(kr.T).T

  '''#bandpass filter
  b, a = signal.butter(2, 0.4, 'low', analog=False)
  data_2_subs_t = signal.filtfilt(b, a, data_2_subs_t, axis=2)'''

  final_t = np.array([])
  for j in range(0 ,data_2_subs_t.shape[0]):
      data_trial=data_2_subs_t[j,:,:].T
      #data_trial.shape

      data_trial_s1=data_trial[0:int(data_2_subs_t.shape[2]/3),:]
      #print(data_trial_s1.shape)
      data_trial_s2=data_trial[int(data_2_subs_t.shape[2]/3):2*int(data_2_subs_t.shape[2]/3),:]
      #print(data_trial_s2.shape)
      data_trial_s3=data_trial[2*int(data_2_subs_t.shape[2]/3):3*int(data_2_subs_t.shape[2]/3),:]
      #print(data_trial_s3.shape)

      #AR Coefficients
      #from statsmodels.datasets.sunspots import load
      #data = load()
      ARFV=np.array([])

      for i in range(0, data_2_subs_t.shape[1]):
          rho1, sigma1 = sm.regression.linear_model.burg(data_trial_s1[:,i], order=2)
          rho2, sigma2 = sm.regression.linear_model.burg(data_trial_s2[:,i], order=2)
          rho3, sigma3 = sm.regression.linear_model.burg(data_trial_s3[:,i], order=2)
          ARFV=np.append(ARFV, (rho1, rho2, rho3))

      #print(ARFV) 

      HWDFV=np.array([])
      for i in range(0, data_2_subs_t.shape[1]):
          (cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          HWDFV=np.append(HWDFV, cA)

      #Spectral Power estimates
      SPFV=np.array([])
      for i in range(0, data_2_subs_t.shape[1]):
          f1, Pxx_den1 = signal.welch(data_trial_s1[:,i], int(data_2_subs_t.shape[2]/3))
          f2, Pxx_den2 = signal.welch(data_trial_s2[:,i], int(data_2_subs_t.shape[2]/3))
          f3, Pxx_den3 = signal.welch(data_trial_s3[:,i], int(data_2_subs_t.shape[2]/3))
          SPFV=np.append(SPFV, (Pxx_den1, Pxx_den2, Pxx_den3))

      #Concatenaton of All the feature vectors
      concated=np.concatenate((ARFV, HWDFV, SPFV), axis=None)
      concated=np.reshape(concated, (-1, 1))
      if j==0:
          final_t=concated
      else:
          final_t= np.hstack((final_t, concated))
      # print(j)
  print(final_t.shape)

  final_t=final_t.T
  final_t.shape

  eegData_t=np.rollaxis(data_2_subs_t, 0, 3)
  eegData_t.shape


  # Subband Information Quantity
  # delta (0.5–4 Hz)
  eegData_delta_t = eeg.filt_data(eegData_t, 0.5, 4, fs)
  ShannonRes_delta_t = eeg.shannonEntropy(eegData_delta_t, bin_min=-200, bin_max=200, binWidth=2)
  # theta (4–8 Hz)
  eegData_theta_t = eeg.filt_data(eegData_t, 4, 8, fs)
  ShannonRes_theta_t = eeg.shannonEntropy(eegData_theta_t, bin_min=-200, bin_max=200, binWidth=2)
  # alpha (8–12 Hz)
  eegData_alpha_t = eeg.filt_data(eegData_t, 8, 12, fs)
  ShannonRes_alpha_t = eeg.shannonEntropy(eegData_alpha_t, bin_min=-200, bin_max=200, binWidth=2)
  # beta (12–30 Hz)
  eegData_beta_t = eeg.filt_data(eegData_t, 12, 30, fs)
  ShannonRes_beta_t = eeg.shannonEntropy(eegData_beta_t, bin_min=-200, bin_max=200, binWidth=2)
  # gamma (30–100 Hz)
  eegData_gamma_t = eeg.filt_data(eegData_t, 30, 80, fs)
  ShannonRes_gamma_t = eeg.shannonEntropy(eegData_gamma_t, bin_min=-200, bin_max=200, binWidth=2)

  # Hjorth Mobility
  # Hjorth Complexity
  HjorthMob_t, HjorthComp_t = eeg.hjorthParameters(eegData_t)


  # Median Frequency
  medianFreqRes_t = eeg.medianFreq(eegData_t,fs)

  # Standard Deviation
  std_res_t = eeg.eegStd(eegData_t)

  # Regularity (burst-suppression)
  regularity_res_t = eeg.eegRegularity(eegData_t,fs)

  # Spikes
  minNumSamples = int(70*fs/1000)
  spikeNum_res_t = eeg.spikeNum(eegData_t,minNumSamples)


  # Sharp spike
  sharpSpike_res_t = eeg.shortSpikeNum(eegData_t,minNumSamples)
  #REMPMVPOMMPOMPOMPOMPOMOMPOMPOM
  concated_t=np.concatenate(( ShannonRes_delta_t.T, ShannonRes_theta_t.T, ShannonRes_alpha_t.T, ShannonRes_beta_t.T, ShannonRes_gamma_t.T, HjorthMob_t.T, HjorthComp_t.T, medianFreqRes_t.T, std_res_t.T, regularity_res_t.T, spikeNum_res_t.T, sharpSpike_res_t.T), axis=1)

  final_t=np.hstack((final_t, concated_t))

  # δ band Power
  bandPwr_delta_t = eeg.bandPower(eegData_t, 0.5, 4, fs)
  #too large
  # θ band Power
  bandPwr_theta_t = eeg.bandPower(eegData_t, 4, 8, fs)
  #too large
  # α band Power
  bandPwr_alpha_t = eeg.bandPower(eegData_t, 8, 12, fs)
  #too large
  # β band Power
  bandPwr_beta_t = eeg.bandPower(eegData_t, 12, 30, fs)
  #too large
  # γ band Power
  bandPwr_gamma_t = eeg.bandPower(eegData_t, 30, 80, fs)

  concated_n_t= bandPwr_gamma_t.T
  final_t=np.hstack((final_t, concated_n_t))
  #final_t.shape

  HTFV_t=np.array([])
  for j in range(0, eegData_t.shape[2]):
    eegData_temp=eegData_t[:,:,j]
    HTFV_temp=np.array([])
    for i in range(0, eegData.shape[0]):
      HTFV_temp=np.append(HTFV_temp, np.imag(hilbert(eegData_temp[i,:])))
    if(j==0):
      HTFV_t=HTFV_temp
    else:
      HTFV_t=np.vstack((HTFV_t, HTFV_temp))


    # print(j)

  final_t=np.hstack((final_t, HTFV_t))
  final_t.shape

  list_rand=[]
  for i in range(0,final.shape[1]):
    list_rand.append("c"+str(i))
  len(list_rand)
  df_train = pd.DataFrame(final, columns = list_rand)
  df_test = pd.DataFrame(final_t, columns = list_rand)

  return df_train, df_test

def NIKHIL_GIVE_NEW(args):
    data_ib=np.load('./data/'+args.dataset+'_epochs.npz')
    data_train_ib = data_ib["X"]
    labels_train_ib = data_ib["y"]
    #500 Tstep
    data_train_ib_500=segment(data_train_ib, segment_length=500)
    print(np.amax(data_train_ib_500))
    print(np.amin(data_train_ib_500))

    segment_length=500
    labels_train_ib_500=np.repeat(labels_train_ib,3000/int(segment_length))

    #1000 Tstep
    data_train_ib_1000=segment(data_train_ib, segment_length=1000)
    segment_length=1000
    labels_train_ib_1000=np.repeat(labels_train_ib,3000/int(segment_length))

    #1500 Tstep
    data_train_ib_1500=segment(data_train_ib, segment_length=1500)
    segment_length=1500
    labels_train_ib_1500=np.repeat(labels_train_ib,3000/int(segment_length))

    #3000 Tstep
    data_train_ib_3000=data_train_ib
    segment_length=3000
    labels_train_ib_3000=labels_train_ib

    training_data=[data_train_ib_500, data_train_ib_1000, data_train_ib_1500, data_train_ib_3000]
    label_data=[labels_train_ib_500, labels_train_ib_1000, labels_train_ib_1500, labels_train_ib_3000]
    segment_length=[500,1000,1500,3000]


    kf3 = KFold(n_splits=3, shuffle=False)
    accd={}

    for i in range(len(training_data)):
        print("iteration "+str(i))
        data_train_loop=training_data[i]
        labels_train_loop=label_data[i]
        #fs=int(segment_length[i]/3)
        acc=[]
        for train_index, test_index in kf3.split(data_train_loop):
            df_train_temp, df_test_temp=createFV_individual(data_train_loop[train_index], data_train_loop[test_index], 1000)
            print(np.amax(df_train_temp.values))
            print(np.amin(df_train_temp.values))
            # Without feature selection check accuracy with Random forest
            estimator = RandomForestClassifier()
            selector = GeneticSelectionCV(
            estimator,
            cv=5,
            verbose=1,
            scoring="accuracy",
            n_population=64,
            crossover_proba=0.5,
            mutation_proba=0.2,
            n_generations=50,
            crossover_independent_proba=0.5,
            mutation_independent_proba=0.05,
            tournament_size=3,
            n_gen_no_change=10,
            caching=True,
            n_jobs=64,)
            selector = selector.fit(df_train_temp.values, labels_train_loop[train_index])
            acc.append(selector.score(df_test_temp.values, labels_train_loop[test_index]))
        accd[str(i)]=sum(acc)/len(acc)

    return accd
    #sb="jc_mot"


