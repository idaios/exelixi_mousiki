## exelixi mousikis

##%matplotlib inline  
import librosa
import librosa.display
import IPython
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import os

data = {}
sr=22000

audio = librosa.util.example_audio_file()

features = {}
mp3list = []
fileList = os.listdir("./")
for f in fileList:
    if f.endswith(".mp3"):
        mp3list.append(f)
mp3list

for file in os.listdir("./"):
    if file.endswith(".mp3"):
        features[file] = {}
        print(file)
        data[file],sr = librosa.load(file, sr=sr)
        y = data[file]
        print('Audio Sampling Rate: '+str(sr)+' samples/sec')
        print('Total Samples: '+str(np.size(data[file])))
        secs=np.size(y)/sr
        print('Audio Length: '+str(secs)+' s')
        ##IPython.display.Audio(y)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        ##features[file]['harmonic'] = y_harmonic
        ##features[file]['percussive'] = y_percussive
        ## tempo
        tempo, beat_frames = librosa.beat.beat_track(y=y_harmonic, sr=sr)
        print('Detected Tempo: '+str(tempo)+ ' beats/min')
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        beat_time_diff=np.ediff1d(beat_times)
        beat_nums = np.arange(1, np.size(beat_times))
        ##features[file]['beat_times'] = beat_times
        ##features[file]['beat_time_diff'] = beat_time_diff
        ##features[file]['beat_nums'] = beat_nums
        ## chroma
        chroma=librosa.feature.chroma_cens(y=y_harmonic, sr=sr)
        ##features[file]['chroma'] = chroma
        ## mfccs
        mfccs = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=13)
        ##features[file]['mfccs'] = mfccs
        ## spectral centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        ##features[file]['cent'] = cent
        ## spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y_harmonic,sr=sr)
        ##features[file]['contrast'] = contrast
        ## rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        ##features[file]['rolloff'] = rolloff
        ## zero crossing rate
        zrate=librosa.feature.zero_crossing_rate(y_harmonic)
        ##features[file]['zrate'] = zrate

        ## features generation
        ## this is a feature that runs on the octave
        features[file]['chroma_mean'] = [np.mean(chroma, axis=1)] ##
        features[file]['chroma_std'] = [np.std(chroma, axis=1)]
        features[file]['mfccs_mean'] = [np.mean(mfccs, axis=1)]
        features[file]['mfccs_std'] = [np.std(mfccs, axis=1)]
        features[file]['cent_mean'] = [np.mean(cent)]
        features[file]['cent_std'] = [np.std(cent)]
        features[file]['cent_skew'] = [scipy.stats.skew(cent, axis=1)[0]]
        features[file]['contrast_mean'] = [np.mean(contrast, axis=1)]
        features[file]['contrast_std'] = [np.std(contrast, axis=1)]
        features[file]['rolloff_mean'] = [np.mean(rolloff)]
        features[file]['rolloff_std'] = [np.std(rolloff)]
        features[file]['rolloff_skew'] = [scipy.stats.skew(rolloff, axis=1)[0]]
        features[file]['zrate_mean'] = [np.mean(zrate)]
        features[file]['zrate_std'] = [np.std(zrate)]
        features[file]['zrate_skew'] = [scipy.stats.skew(zrate, axis=1)[0]]
        features[file]['tempo'] = [tempo]
        

print(features)
f0 = mp3list[0]
features_df = pd.DataFrame()
for key in features[f0].keys():
    print(key)
    for kk in range(len(features[f0][key])):
        name=f"{key}_{kk}"
        features_df[name] = features[f0][key][kk]
        print(name)


print(features[f0]['chroma_mean'])

    
        
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 5)
        ax.set_ylabel("Time difference (s)")
        ax.set_xlabel("Beats")
        g=sns.barplot(beat_nums, beat_time_diff, palette="BuGn_d",ax=ax)
        g=g.set(xticklabels=[])
        plt.savefig(f"beats_{file}.pdf")
##len(data['psarantonis1.mp3'][0])

secs=np.size(y)/sr
print('Audio Length: '+str(secs)+' s')
IPython.display.Audio(audio)


plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
