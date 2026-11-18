import ReadingPLAIDMeta as rpm
import numpy as np
import pandas as pd
from glob import glob
import re
import AugmentationFunctions as aug
import scipy
import os
import soundfile as sf
import time

# PLAID Class Names
plaid_classes = np.array(['Air Conditioner', 'Compact Fluorescent Lamp', 'Fan', 'Fridge', 'Hairdryer',
                            'Heater', 'Incandescent Light Bulb', 'Laptop', 'Microwave', 'Vacuum',
                            'Washing Machine'])


#Reshping Augmented Data arrays
def arr_reshape(data, labels):
    rows = data.shape[0]
    for i in data.shape[2:]:
        rows *= i
    tpose_tuple = list(np.arange(data.ndim))[::-1][:-2]
    tpose_tuple.extend([0, 1])
    tpose_tuple = tuple(tpose_tuple)
    data_reshaped = data.transpose(tpose_tuple).reshape((rows, data.shape[1]))
    labels_duplicated = np.reshape(np.tile(labels, int(rows/data.shape[0])), (-1, 1))
    return data_reshaped, labels_duplicated


plaid_AC_data_total_time_start = time.time()
plaid_AC_labels_total_time_start = time.time()
plaid_RMS_data_total_time_start = time.time()
plaid_RMS_labels_total_time_start = time.time()

#pulling from PLAID

#gets all labels from PLAID
labels = rpm.read_plaid_meta()

#some important parameters
sample_freq = 30000 #original sampling rate
mains_freq = 60
new_sample_freq = 5000 #new sampling frequency
start_buffer_time = 0.05 #the max buffer time before the device switch on (in seconds)
num_rows_start = np.floor(new_sample_freq*start_buffer_time).astype("int") #maximum number of rows allowed before event start
threshold = 1.5 #the minimum threshold to define an event (in Watts)
time_len = 1 #the desired length of each example file (in seconds) after cleaning. Num of data points will then be time_len*new_sample_freq
num_rows_total = np.floor(time_len*new_sample_freq).astype("int") #number of rows allowed in the full file

#numerically sort through files
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
files = sorted(glob("/user/work/xh21734/Intern/Data/PLAID/2017/*"), key=numericalSort) #this is the folder that PLAID data is saved to
good_labels = []
extracted_data = []

for i, file in enumerate(files):

    file_data = pd.read_csv(file, header=None, names=("current", "voltage"))
    #calculate apparent power from each file
    data_df=rpm.apparent_power(file_data, sample_freq, mains_freq)
    #downsample data to set sampling frequency
    data_df = rpm.downsample(data_df, sample_freq, new_sample_freq, axis=0)

    #cut off start of recording if it is more than the start_buffer_time
    data_df, is_bad_file = rpm.event_start(data_df, threshold, num_rows_start, num_rows_total)
    #print(i)
    #check if data_df is not a dataframe, meaning that that file is a bad file and has no suitable events
    if not is_bad_file:
        #repeatedly appending to a list is faster than numpy stack type
        good_labels.append(labels.iloc[i].loc["appliance_type"])
        extracted_data.append(data_df[["apparent_power","current","voltage", "rms_current"]].to_numpy()[np.newaxis, :, :])

plaid_labels_index = []
extracted_data = np.vstack(extracted_data)
good_labels = np.asarray(good_labels)

for i in range(len(plaid_classes)):
    loc = np.where(good_labels == plaid_classes[i])
    loc = np.array(loc).tolist()
    plaid_labels_index.extend(loc)

plaid_labels_index = sum(plaid_labels_index,[])

plaid_good_data = np.take(extracted_data,plaid_labels_index,axis=0)
plaid_good_labels = np.take(good_labels,plaid_labels_index,axis=0)



#saving plaid raw curr data and labels
#data_and_labels = np.hstack((good_labels.reshape((-1, 1)), extracted_data[:, :, 1].astype(np.float32)))
AC_curr_data = plaid_good_data[:, :, 1].astype(np.float32)
RMS_curr_data = plaid_good_data[:, :, 3].astype(np.float32)
save_path = "/user/work/xh21734/Intern/TestData/Basic_Run/"


#np.save((save_path + "PLAID_5000Hz_AC_data.npy"), AC_curr_data)
#np.save((save_path + "PLAID_5000Hz_RMS_data.npy"), RMS_curr_data)
#np.save((save_path + "PLAID_5000Hz_curr_labels.npy"), good_labels)

plaid_AC_data = AC_curr_data
AC_augmented_data = aug.change_mains_freq(60, 50, plaid_AC_data)
AC_augmented_data = aug.window_shift(AC_augmented_data, 10, sample_freq=5000)
AC_augmented_data = aug.scale(AC_augmented_data)

#loading plaid RMS data

plaid_RMS_data = RMS_curr_data
#augmented_data = aug.change_mains_freq(60, 50, plaid_RMS_data) #don't need this for RMS
RMS_augmented_data = aug.window_shift(plaid_RMS_data, 10, sample_freq=5000)
RMS_augmented_data = aug.scale(RMS_augmented_data, num_splits=5)

#saving augmented data

PLAID_AC_reshaped_data, PLAID_AC_reshaped_labels = arr_reshape(AC_augmented_data, plaid_good_labels)
PLAID_RMS_reshaped_data, PLAID_RMS_reshaped_labels = arr_reshape(RMS_augmented_data, plaid_good_labels)

print("PLAID data shapes:")
print(str(len(PLAID_AC_reshaped_data))+"x"+str(len(PLAID_AC_reshaped_data[0])))
#print("PLAID AC labels shape:")
print(len(PLAID_AC_reshaped_labels))
#print("PLAID RMS data shape:")
print(str(len(PLAID_RMS_reshaped_data))+"x"+str(len(PLAID_RMS_reshaped_data[0])))
#print("PLAID RMS labels shape:")
print(len(PLAID_RMS_reshaped_labels))



print("PLAID total application time:")

np.save((save_path  + "/PLAID_5000Hz_AC_AUGMENTED_data.npy"), np.array(PLAID_AC_reshaped_data))
plaid_AC_data_total_time_end = time.time()
print(str(plaid_AC_data_total_time_end-plaid_AC_data_total_time_start))
np.save((save_path  + "/PLAID_5000Hz_AC_labels.npy"), np.array(PLAID_AC_reshaped_labels))    
plaid_AC_labels_total_time_end = time.time()
print(str(plaid_AC_labels_total_time_end-plaid_AC_labels_total_time_start))
np.save((save_path  + "/PLAID_5000Hz_RMS_AUGMENTED_data.npy"), np.array(PLAID_RMS_reshaped_data))
plaid_RMS_data_total_time_end = time.time()
print(str(plaid_RMS_data_total_time_end-plaid_RMS_data_total_time_start))
np.save((save_path  + "/PLAID_5000Hz_RMS_labels.npy"), np.array(PLAID_RMS_reshaped_labels))
plaid_RMS_labels_total_time_end = time.time()
print(str(plaid_RMS_labels_total_time_end-plaid_RMS_labels_total_time_start))
