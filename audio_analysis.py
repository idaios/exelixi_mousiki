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

#audio = librosa.util.example_audio_file()

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
        features[file]['chroma_mean'] = np.mean(chroma, axis=1) ##
        features[file]['chroma_std'] = np.std(chroma, axis=1)
        features[file]['mfccs_mean'] = np.mean(mfccs, axis=1)
        features[file]['mfccs_std'] = np.std(mfccs, axis=1)
        features[file]['cent_mean'] = np.mean(cent)
        features[file]['cent_std'] = np.std(cent)
        features[file]['cent_skew'] = scipy.stats.skew(cent, axis=1)[0]
        features[file]['contrast_mean'] = np.mean(contrast, axis=1)
        features[file]['contrast_std'] = np.std(contrast, axis=1)
        features[file]['rolloff_mean'] = np.mean(rolloff)
        features[file]['rolloff_std'] = np.std(rolloff)
        features[file]['rolloff_skew'] = scipy.stats.skew(rolloff, axis=1)[0]
        features[file]['zrate_mean'] = np.mean(zrate)
        features[file]['zrate_std'] = np.std(zrate)
        features[file]['zrate_skew'] = scipy.stats.skew(zrate, axis=1)[0]
        features[file]['tempo'] = tempo
        
features_df = pd.DataFrame()
for song in features.keys():
    f0 = song
    newrow={}
    for key in features[f0].keys():
        print(key)
        if isinstance(features[f0][key], np.ndarray):
            for kk in range(len(features[f0][key])):
                name=f"{key}_{kk}"
                print(features[f0][key][kk])
                newrow[name] = features[f0][key][kk]
                print(features[f0][key][kk])
        else:
            name = key
            newrow[name] = features[f0][key]
    features_df = pd.concat([features_df, pd.DataFrame([newrow])])
    
print(features_df)


from sklearn.decomposition import PCA
import plotly.express as px

pca = PCA()
df = features_df
df.index = features.keys()
df.index
singer=[ind[0:4] for ind in df.index]
##singer=[ind for ind in df.index]
singer
##singer=['skordalos', 'psarantonis', 'psarantonis', 'psarantonis', 'skordalos', 'psarantonis', 'skordalos', 'skordalos']
singercol=[]
for s in singer:
    if s is 'skordalos':
        singercol.append('red')
    elif s is 'psarantonis':
        singercol.append('blue')
        
df_normalized=(df - df.mean()) / df.std()

components = pca.fit_transform(df_normalized)
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}

fig = px.scatter_matrix(
    components,
    labels=labels,
    dimensions=range(4),
    color=singer
)
fig.update_traces(diagonal_visible=False)
fig.show()


