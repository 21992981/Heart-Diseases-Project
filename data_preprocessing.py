# Importing libaries
import pandas as pd
import glob
import os
import re
from scipy.io import wavfile
import numpy as np
import math

# Find all csv files in folder and read
def find_str(pattern, text):
    match = re.search(pattern, text)
    if match:
        val = match.group(1)
        return val
    else:
        return None

cwd = os.getcwd()
directory = cwd + '/datasets/old_data/hastaverileri/'
csv_files = glob.glob(os.path.join(directory, '**', '*.csv'), recursive=True)
csv_files = sorted(csv_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))

dfs = []
for file in csv_files:
    patient = int(re.search(r'/(\d+)\.csv$', file).group(1))
    df = pd.read_csv(file)
    df['patient'] = patient
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
combined_df['datetime'] = pd.to_datetime(combined_df['Date'] + ' ' + combined_df['Time'], format='%d.%m.%Y %H:%M:%S')
timeseries_df = combined_df.drop(columns=['Date', 'Time'])

# Read wav files
cwd = os.getcwd()
directory = cwd + '/datasets/old_data/hastaverileri/'
wav_files = glob.glob(os.path.join(directory, '**', '*.wav'), recursive=True)

data_list = []
patient_list = []

for wav in wav_files:
    patient = int(re.search(r'/(\d+)\.wav$', wav).group(1))
    patient_list.append(patient)
    samplerate, data = wavfile.read(wav)
    data_list.append(data)

wav_files = sorted(wav_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
wav_df = pd.DataFrame(columns=['patient', 'rate_min', 'rate_max', 'rate_std', 'rate_avg'])

# Reduce audio sample rate and synchronize with ECG data
df_list = list()
for wav in wav_files:
    patient = int(re.search(r'/(\d+)\.wav$', wav).group(1))
    samplerate, data = wavfile.read(wav)
    resampler = math.floor(len(data) / len(timeseries_df[timeseries_df['patient']==patient]))
    del_val = len(data) - (resampler*len(timeseries_df[timeseries_df['patient']==patient]))
    del_val = math.ceil(len(data)/del_val)
    indices_to_delete = np.arange((del_val-1), len(data), del_val)

    modified_array = np.delete(data, indices_to_delete)
    num_chunks = len(modified_array) // resampler
    reshaped_array = modified_array[:num_chunks * resampler].reshape(num_chunks, resampler)
    averages = reshaped_array.mean(axis=1)
    num_chunks
    descriptive_stats = []

    for chunk in reshaped_array:
        stats = {
            'patient':patient, 
            'rate_min': np.min(chunk),
            'rate_max': np.max(chunk),
            'rate_std': np.std(chunk),
            'rate_avg': np.mean(chunk)
        }
        descriptive_stats.append(stats)
    df_list.append(pd.DataFrame(descriptive_stats))

# Merge timeseries data and wav data
wav_df = pd.concat(df_list)

timeseries_df['rate_avg'] = wav_df['rate_avg'].values
timeseries_df['rate_std'] = wav_df['rate_std'].values
timeseries_df['rate_min'] = wav_df['rate_min'].values
timeseries_df['rate_max'] = wav_df['rate_max'].values

# Save the data
timeseries_df.to_csv('datasets/main.csv')
