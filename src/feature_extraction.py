import numpy as np
import pandas as pd
import os

sampling_rate = 500 # in Hz
window_seconds = 0.15 # in seconds
window_size = int(window_seconds * sampling_rate)

# mean absolute value
def mav(data):
    return np.mean(np.abs(data), axis=0)

# root mean square
def rms(data):
    return np.sqrt(np.mean(data**2,axis=0))  
    
# wavelength
def wavelength(data):
    return np.sum(np.abs(np.diff(data)), axis=0)
    
# zero crossing rate
def zcr(data):
    return np.sum(np.diff(np.sign(data), axis=0) != 0, axis=0) / (len(data)-1)

# variance
def var(data):
    return np.var(data, axis=0)

# absolute difference
def abs_diffs_signal(data):
    return np.sum(np.abs(np.diff(data,axis=0)),axis=0)

# mean frequency
def mean_freq(data, fs=500):
    freqs = np.fft.rfftfreq(len(data), d=1/fs)
    spectrum = np.abs(np.fft.rfft(data))**2
    return np.sum(freqs * spectrum) / np.sum(spectrum)

# median frequency
def median_freq(data, fs=500):
    freqs = np.fft.rfftfreq(len(data), d=1/fs)
    spectrum = np.abs(np.fft.rfft(data))**2
    cumulative = np.cumsum(spectrum)
    total = cumulative[-1]
    med_idx = np.searchsorted(cumulative, total / 2)
    return freqs[med_idx]

# peak frequency
def peak_freq(data, fs=500):
    freqs = np.fft.rfftfreq(len(data), d=1/fs)
    spectrum = np.abs(np.fft.rfft(data))**2
    peak_idx = np.argmax(spectrum)
    return freqs[peak_idx]

# shannon entropy
def shannon_entropy(signal, num_bins=30):
    hist, bin_edges = np.histogram(signal, bins=num_bins, density=True)
    prob = hist * np.diff(bin_edges)  

    prob = prob[prob > 0]

    entropy = -np.sum(prob * np.log2(prob))
    return entropy

# integrated emg
def iemg(signal):
    return np.sum(np.abs(signal))

# make df from data path
def make_df(data_path, exclude, rectify=False, smooth=False):
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

            features_df = sample_df_grouped.agg(['min', 'max', mav, rms, wavelength, var, abs_diffs_signal, shannon_entropy, iemg])

            df = pd.concat([df, features_df])
    
    df.reset_index(inplace=True)
    df['substance'] = df['substance'].map(class_map)
    return df, class_map