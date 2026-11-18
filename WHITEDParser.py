import reading_plaid_meta as rpm
import numpy as np
import pandas as pd
from glob import glob
import re
import AugmentationFunctions as aug
import scipy
import os
import soundfile as sf
import time

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

#WHITED Class Names
whited_classes = np.array(['AC', 'CFL', 'Fan', 'Fridge', 'HairDryer',
                            'Heater', 'LightBulb', 'Laptop', 'Microwave', 'VacuumCleaner',
                            'WashingMachine'])


#loading in WHITED data

whited_AC_data_total_time_start = time.time()
whited_AC_labels_total_time_start = time.time()
whited_RMS_data_total_time_start = time.time()
whited_RMS_labels_total_time_start = time.time()

new_samplerate = 5000 #new sampling frequency
start_buffer_time = 0.05 #the max buffer time before the device switch on (in seconds)
num_rows_start = np.floor(new_samplerate*start_buffer_time).astype("int") #maximum number of rows allowed before event start
threshold = 1.5 #the minimum threshold to define an event (in Watts)

#This needs to be more than the min length of 1 second, so that after augmenting the 
# mains freq from 50Hz to 60Hz, there are enough data points to match the 5000 data points in PLAID
time_len = 1.2 #the desired length of each example file (in seconds) after cleaning. Num of data points will then be time_len*new_sample_freq
num_rows_total = np.floor(time_len*new_samplerate).astype("int") #number of rows allowed in the full file

#create dictionaries of current and voltage factors, and mains frequencies
location_dict = {
    "r1": ["Germany", 50],
    "r2": ["Germany", 50],
    "r3": ["Germany", 50],
    "r4": ["Indonesia", 50],
    "r5": ["Indonesia", 50],
    "r6": ["Austria", 50],
    "r7": ["Germany", 50],
    "r8": ["Canada", 60]
}

calibration_dict ={
    "MK1": [1033.64, 61.4835], #volt factor then current factor
    "MK2": [861.15, 60.200],
    "MK3": [988.926, 60.9562]
}

file_len = 224910 #length of files in WHITED dataset
samplerate = 44100 #in Hz

mains=[]
for key, val in location_dict.items():
    mains.append(val[1])
cut_beginning = int((samplerate/min(mains))-1) #number of datapoints to be removed from beginning of file based on lowest mains frequency
new_len = file_len-cut_beginning
#print(cut_beginning)
#function to get all files the same length given the different mains frequencies
def chop_length(data, new_len):
    if len(data)==new_len:
        return data
    else:
        chop=len(data)-new_len
        return data.iloc[chop:,:]

def apparent_power(data, volt_factor, current_factor, samples):
    data["voltage"] = data["voltage"]*volt_factor #converts to normal units of voltage
    data["current"] = data["current"]*current_factor #converts to normal units of current
    data["rms_voltage"] = ((abs(data["voltage"])**2).rolling(samples).mean())**0.5
    data["rms_current"] = ((abs(data["current"])**2).rolling(samples).mean())**0.5
    data["apparent_power"] = data["rms_voltage"]*data["rms_current"]
    data = data.dropna()
    return data

def whited_labels(file_name): #create a row to be added to np array. Columns= types, brand, region, calibrations, time
    file_string=str(file_name).split("WHITED/")[1]
    types = file_string.split("_")[0]
    brand = file_string.split("_")[1]
    region = file_string.split("_")[2]
    calibrations = file_string.split("_")[3]
    time = (file_string.split("_")[4]).split(".")[0]
    arr_row = np.array([types, brand, region, calibrations, time])
    return arr_row

label_arr = [] #empty array to be filled with 5 columns of label information
curr_vol = []


#os.chdir("/user/work/xh21734/Intern/Data/WHITED/")
for file in glob("/user/work/xh21734/Intern/Data/WHITED/*.flac"): #WHITED folder containing .flac files
    dataset = sf.SoundFile(file)
    data = pd.DataFrame(dataset.read(), columns = ("voltage", "current")) #data contains voltage in 1st column, and current in 2nd column
    dataset.close() #just being good and closing the file
    label_row = whited_labels(file) #creating a row of info for each file
    #using that row info to get values from the dictionaries
    volt_factor = (calibration_dict[label_row[3]])[0]
    current_factor = (calibration_dict[label_row[3]])[1]
    if label_row[2]!="r9": #a random bad piece of data needs to be ommitted
        mains_freq = (location_dict[label_row[2]])[1]
        samples = int(samplerate/mains_freq)
        #calculating apparent power
        data = apparent_power(data, volt_factor, current_factor, samples)
        #chopping to same length
        #data = chop_length(data, new_len)
        #downsampling
        data = rpm.downsample(data, samplerate, new_samplerate)
        #taking 1 sec window around where the event occurs
        data, is_bad_file = rpm.event_start(data, threshold, num_rows_start, num_rows_total)

        if not is_bad_file:
            label_arr.append(label_row)
            curr_vol.append(data[["apparent_power","current","voltage", "rms_current"]].to_numpy()[np.newaxis, :, :])


whited_labels_index = []
label_arr = np.vstack(label_arr)
curr_vol = np.vstack(curr_vol)

for i in range(len(whited_classes)):
    loc = np.where(label_arr[:, 0] == whited_classes[i])
    loc = np.array(loc).tolist()
    whited_labels_index.extend(loc)

whited_labels_index = sum(whited_labels_index,[])

whited_good_data = np.take(curr_vol,whited_labels_index,axis=0)
whited_good_labels = np.take(label_arr,whited_labels_index,axis=0)


#printing number of each appliance type
#overlapping classes between the two datasets and their respective label names

#print(pd.DataFrame(whited_good_labels[:, 0], columns=["appliance"])["appliance"].value_counts())

#removing mosquito repellant and saving data
whited_mask = np.argwhere(whited_good_labels[:, 0] == "MosquitoRepellent")
whited_AC_data = whited_good_data[:, :, 1].astype(np.float32)
whited_AC_data = np.delete(whited_AC_data, whited_mask, axis=0)

whited_RMS_data = whited_good_data[:, :5000, 3].astype(np.float32)
whited_RMS_data = np.delete(whited_RMS_data, whited_mask, axis=0)
whited_RMS_labels = np.delete(whited_good_labels[:, 0], whited_mask, axis=0)
#np.save((save_path + "WHITED_5000Hz_AC_data.npy"), whited_AC_data)
#np.save((save_path + "WHITED_5000Hz_RMS_data.npy"), whited_RMS_data)
#np.save((save_path + "WHITED_5000Hz_RMS_labels.npy"), whited_RMS_labels)
#loading whited data and labels
save_path = "/user/work/xh21734/Intern/TestData/Basic_Run/"

#whited_labels = np.load((save_path + "WHITED_5000Hz_labels.npy"), allow_pickle=True)
def change_whited_freq(old_freq, new_freq, data):
    num_samples = data.shape[1]
    if new_freq<old_freq: #if going to 60Hz from 50Hz
        new_num_samples = int((num_samples*old_freq)/new_freq)
        new_freq_data = scipy.signal.resample(data, new_num_samples, axis=1)[:,:num_samples]

    else: #if going from 50Hz to 60Hz
        downsample_ratio = round((new_freq/old_freq),1) #save to 1 decimal place(1.2)
        new_freq_data = scipy.signal.resample(data, num_samples*10, axis=1)
        #uses a downsample ratio of 12 on 10x data, is the same as an exact
        # downsample ratio of 1.2
        new_freq_data = new_freq_data[:,::int(downsample_ratio*10)]

    return new_freq_data
#selectively changing mains frequency of whited data
num_rows = 5000
#sorting whited data based on r value
labels_df = pd.DataFrame(np.delete(whited_good_labels, whited_mask, axis=0), columns=["app", "brand", "code", "fact", "time"])
whited_df = pd.DataFrame(whited_AC_data).set_index(labels_df["code"]).sort_index() #sort from r1 to r8
whited_50Hz = whited_AC_data[labels_df["code"] != "r8"]
whited_50Hz_labels = labels_df[labels_df["code"] != "r8"]["app"].to_numpy()
whited_50Hz_aug = change_whited_freq(50, 60, whited_50Hz)[:, :num_rows]

whited_60Hz = whited_AC_data[whited_df.index == "r8"]
whited_60Hz_labels = labels_df[labels_df["code"] == "r8"]["app"].to_numpy()
whited_60Hz_aug = change_whited_freq(60, 50, whited_60Hz)[:, :num_rows]

aug_arr = np.vstack((whited_50Hz_aug, whited_60Hz_aug))
whited_labels_sorted = np.vstack((whited_50Hz_labels.reshape((-1, 1)), whited_60Hz_labels.reshape(-1, 1)))

#reordering the original data so that it matches that of the augmented data
temp_whited_reordered = np.vstack((whited_AC_data[labels_df["code"] != "r8", :num_rows], whited_AC_data[labels_df["code"] == "r8", :num_rows]))
whited_aug1_data = np.stack((temp_whited_reordered, aug_arr), axis=-1)
#saving whited labels
#np.save((save_path + "WHITED_5000Hz_AC_labels.npy"), whited_labels_sorted)
#augmenting whited data
#augmented_whited_AC_data = aug.change_mains_freq(60, 50, whited_AC_data)
augmented_whited_AC_data = aug.window_shift(whited_aug1_data, 10, sample_freq=5000)
augmented_whited_AC_data = aug.scale(augmented_whited_AC_data).astype(np.float32)
#saving whited augmented data
#np.save((save_path + "WHITED_5000Hz_AC_AUGMENTED_data.npy"), augmented_whited_AC_data)
#augmenting whited RMS data
augmented_whited_RMS_data = aug.window_shift(whited_RMS_data, 10, sample_freq=5000)
augmented_whited_RMS_data = aug.scale(augmented_whited_RMS_data).astype(np.float32)
#saving RMS augmented data
#np.save((save_path + "WHITED_5000Hz_RMS_AUGMENTED_data.npy"), augmented_whited_RMS_data)


#saving augmented data

WHITED_AC_reshaped_data, WHITED_AC_reshaped_labels = arr_reshape(augmented_whited_AC_data, whited_labels_sorted)


WHITED_RMS_reshaped_data, WHITED_RMS_reshaped_labels = arr_reshape(augmented_whited_RMS_data, whited_RMS_labels)



print("WHITED data shapes:")
print(str(len(WHITED_AC_reshaped_data))+"x"+str(len(WHITED_AC_reshaped_data[0])))
#print("WHITED AC labels shape:")
print(len(WHITED_AC_reshaped_labels))
#print("WHITED RMS data shape:")
print(str(len(WHITED_RMS_reshaped_data))+"x"+str(len(WHITED_RMS_reshaped_data[0])))
#print("WHITED RMS labels shape:")
print(len(WHITED_RMS_reshaped_labels))



print("WHITED total application time:")

np.save((save_path  + "/WHITED_5000Hz_AC_AUGMENTED_data.npy"), np.array(WHITED_AC_reshaped_data))
whited_AC_data_total_time_end = time.time()
print(str(whited_AC_data_total_time_end-whited_AC_data_total_time_start))
np.save((save_path  + "/WHITED_5000Hz_AC_labels.npy"), np.array(WHITED_AC_reshaped_labels))    
whited_AC_labels_total_time_end = time.time()
print(str(whited_AC_labels_total_time_end-whited_AC_labels_total_time_start))
np.save((save_path  + "/WHITED_5000Hz_RMS_AUGMENTED_data.npy"), np.array(WHITED_RMS_reshaped_data))
whited_RMS_data_total_time_end = time.time()
print(str(whited_RMS_data_total_time_end-whited_RMS_data_total_time_start))
np.save((save_path  + "/WHITED_5000Hz_RMS_labels.npy"), np.array(WHITED_RMS_reshaped_labels))
whited_RMS_labels_total_time_end = time.time()
print(str(whited_RMS_labels_total_time_end-whited_RMS_labels_total_time_start))
