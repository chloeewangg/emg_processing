import numpy as np
import pandas as pd
import os

from scipy.signal import welch

sampling_rate = 500 # hz
window_seconds = 0.15 # s
window_size = int(window_seconds * sampling_rate)

# mean absolute value
def mav(data):
    return np.mean(np.abs(data), axis=0)

# root mean square
def rms(data):
    return np.sqrt(np.mean(data**2,axis=0))  
    
# zero crossing rate
def zcr(data):
    return np.sum(np.diff(np.sign(data), axis=0) != 0, axis=0) / (len(data)-1)

# variance
def variance(data):
    return np.var(data, axis=0)

# slope sign change
def ssc(data):
    diff1 = np.diff(data, axis=0)
    diff2 = np.diff(diff1, axis=0)
    ssc = np.sum(((diff1[:-1] * diff1[1:]) < 0) & (np.abs(diff2) >= 1e-6), axis=0)
    return ssc

# absolute difference
def abs_diffs(data):
    return np.sum(np.abs(np.diff(data,axis=0)),axis=0)

# mean frequency
def mean_freq(data, fs=500):
    freqs, psd = compute_psd(data, fs)
    return np.sum(freqs * psd) / np.sum(psd)

# median frequency
def median_freq(data, fs=500):
    freqs, psd = compute_psd(data, fs)
    cumulative = np.cumsum(psd)
    total = cumulative[-1]
    med_idx = np.searchsorted(cumulative, total / 2)
    return freqs[med_idx]

# peak frequency
def peak_freq(data, fs=500):
    freqs, psd = compute_psd(data, fs)
    peak_idx = np.argmax(psd)
    return freqs[peak_idx]

# total power (integrated PSD)
def total_power(data, fs=500):
    freqs, psd = compute_psd(data, fs)
    df = freqs[1] - freqs[0]  # frequency resolution
    return np.sum(psd) * df

# bandwidth (variance around mean frequency)
def bandwidth(data, fs=500):
    freqs, psd = compute_psd(data, fs)
    mean_f = np.sum(freqs * psd) / np.sum(psd)
    return np.sqrt(np.sum(((freqs - mean_f) ** 2) * psd) / np.sum(psd))

# shannon entropy
def shannon_entropy(data, num_bins=30):
    hist, bin_edges = np.histogram(data, bins=num_bins, density=True)
    prob = hist * np.diff(bin_edges)  

    prob = prob[prob > 0]

    entropy = -np.sum(prob * np.log2(prob))
    return entropy

# integrated emg
def iemg(data):
    return np.sum(np.abs(data))

# helper: compute Welch PSD
def compute_psd(data, fs=500):
    nperseg = min(500, len(data))
    freqs, psd = welch(data, fs=fs, nperseg=nperseg, window='hamming')
    return freqs, psd

# make df from data path
def make_df(data_path, exclude, rectify=False, smooth=False, all_features=True):
    '''
    Makes feature dataframe from signals in data path.
    
    args:
        data_path (str): path to data
        exclude (list): list of classes to exclude
        rectify (bool): whether to rectify the data
        smooth (bool): whether to smooth the data
    returns:
        df (pd.DataFrame): dataframe with features
        class_map (dict): dictionary of class names to indices
    '''
    df = pd.DataFrame()

    class_map = {}
    i = 0
    
    for class_name in os.listdir(data_path):
        if class_name in exclude:
            continue
            
        class_path = os.path.join(data_path, class_name)
        class_str = class_name.rstrip('.txt')
        parts = class_str.split()
        volume = parts[-1]
        substance = ' '.join(parts[:-1])
        
        for sample_name in os.listdir(class_path):
            sample_df = pd.read_csv(os.path.join(data_path, class_name, sample_name), delimiter = ",", header = None)

            if rectify:
                sample_df = sample_df.abs()
                
                if smooth:
                    sample_df = sample_df.rolling(window=window_size).apply(rms, raw=True)
                    sample_df = sample_df.dropna()

            sample_df.columns = [i+1 for i in range(sample_df.shape[1])]
            if substance not in class_map:
                class_map[substance] = i
                i += 1
            
            sample_df['substance'] = substance
            sample_df['volume'] = volume
            sample_df_grouped = sample_df.groupby(['substance', 'volume'])

            if all_features:
                features_df = sample_df_grouped.agg(['min', 'max', mav, rms, zcr, variance, ssc, abs_diffs, mean_freq, total_power, iemg, shannon_entropy, median_freq, peak_freq, bandwidth])
            else:
                features_df = sample_df_grouped.agg(['min', ssc, abs_diffs, shannon_entropy, median_freq, peak_freq, bandwidth])

            df = pd.concat([df, features_df])
    
    df.reset_index(inplace=True)
    df['substance'] = df['substance'].map(class_map)
    df['volume'] = df['volume'].astype(int)
    return df, class_map