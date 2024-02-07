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
import re
import random
data = {}
sr=22000

#audio = librosa.util.example_audio_file()
pattern = re.compile(r'psarantonis|skordalos|hpeirwtika')
pattern2 = re.compile(r'skordalos|psarantonis')


features = {}
mp3list = []
fileList = os.listdir("./")
for f in fileList:
    if f.endswith(".mp3") and pattern.match(f):
        mp3list.append(f)

mp3list


for file in mp3list:
    if file.endswith(".mp3"):
        features[file] = {}
        print(file)
        audio,sr = librosa.load(file, sr=sr)
        y = audio
        print('Audio Sampling Rate: '+str(sr)+' samples/sec')
        print('Total Samples: '+str(np.size(audio)))
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
        ## chroma
        chroma=librosa.feature.chroma_cens(y=y_harmonic, sr=sr)
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

ll=list(features.keys())

keysToUse=[]
features_df = pd.DataFrame()
for song in ll:
    if not pattern2.match(song):
        continue
    keysToUse.append(song)
    print(song)
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
    if s == 'skordalos':
        singercol.append('red')
    elif s == 'psarantonis':
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


fig2 = px.scatter(components, x=0, y=1, color=singer, text=df.index)
fig2.show()


############### play with one song

### https://www.educative.io/answers/how-to-perform-note-tracking-in-librosa
### Get the notes and their duration


import numpy as np
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_path = '_static/playback-thumbnail.png'
import librosa
# We'll need IPython.display's Audio widget
from IPython.display import Audio
# We'll also use `mir_eval` to synthesize a signal for us
import mir_eval.sonify


allnotes={}
alldurations={}
for audio_file in mp3list:
    # Loading the audio file 
    y, sr = librosa.load(audio_file)
    # Extracting the chroma features and onsets
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)
    onset_frames = librosa.onset.onset_detect(y=y_harmonic, sr=sr)
    first = True
    notes = []
    mypitch=[]
    mydurations=[]
    for onset in onset_frames:
        chroma_at_onset = chroma[:, onset]
        note_pitch = chroma_at_onset.argmax()
        # For all other notes
        if not first:
            note_duration = librosa.frames_to_time(onset, sr=sr)
            notes.append((note_pitch,onset, note_duration - prev_note_duration))
            prev_note_duration = note_duration
            # For the first note
        else:
            prev_note_duration = librosa.frames_to_time(onset, sr=sr)
            first = False
    print("Note pitch \t Onset frame \t Note duration")
    for entry in notes:
        print(entry[0],'\t\t',entry[1],'\t\t',entry[2])
        mypitch.append(entry[0])
        mydurations.append(entry[2])
    allnotes[audio_file] = mypitch
    alldurations[audio_file] = mydurations
    len(notes)

notes[0]

[len(i) for i in allnotes.values()]
allnotes['psarantonis1.mp3'][:20]

for i in allnotes.keys():
    print(f">{i}")
    for j in range(len(allnotes[i])):
        print(chr(allnotes[i][j]+65), end="")
    print()
    



## APPROACH 3 without onset but split time
    
############### play with one song

### https://www.educative.io/answers/how-to-perform-note-tracking-in-librosa
### Get the notes and their duration


import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
# sphinx_gallery_thumbnail_path = '_static/playback-thumbnail.png'
import librosa
# We'll need IPython.display's Audio widget
from IPython.display import Audio
# We'll also use `mir_eval` to synthesize a signal for us
import mir_eval.sonify

audio_file

allnotes={}
alldurations={}
nres = 1000
resolution = 24
for audio_file in mp3list:
    # Loading the audio file 
    y, sr = librosa.load(audio_file)
    # Extracting the chroma features and onsets
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    ##   totalDuration = librosa.get_duration(y=y, sr=sr)
    ##  totalDuration
    ##  len(y_harmonic)
    ##  step = len(y_harmonic)//nres - 1
    ##  j = step
    ##    while j < len(y_harmonic):

    chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr, n_fft=2048, hop_length=256, win_length=2048, n_chroma=24)
    onset_frames = librosa.onset.onset_detect(y=y_harmonic, sr=sr)
    first = True
    notes = []
    mypitch=[]
    mydurations=[]
    #    chroma.shape
    #    len(onset_frames)
    for onset in range(chroma.shape[1]): #onset_frames:
        chroma_at_onset = chroma[:, onset]
        note_pitch = chroma_at_onset.argmax()
        # For all other notes
        if not first:
            note_duration = librosa.frames_to_time(onset, sr=sr)
            notes.append((note_pitch,onset, note_duration - prev_note_duration))
            prev_note_duration = note_duration
            # For the first note
        else:
            prev_note_duration = librosa.frames_to_time(onset, sr=sr)
            first = False
    print("Note pitch \t Onset frame \t Note duration")
    for entry in notes:
        print(entry[0],'\t\t',entry[1],'\t\t',entry[2])
        mypitch.append(entry[0])
        mydurations.append(entry[2])
    allnotes[audio_file] = mypitch
    alldurations[audio_file] = mydurations
    len(notes)

notes[0]

[len(i) for i in allnotes.values()]
allnotes['psarantonis1.mp3'][:20]

countNotesDif=[]
countNotesDifFrequencies=[]
corfactor = 65 + resolution
with open("songs.fa", "w") as outf:
    for i in allnotes.keys():
        cur_note = ""
        prev_note = chr(allnotes[i][0]+65)
        outf.write(f">{i}\n")
        jj = 0
        for j in range(1,len(allnotes[i])):
            dif = allnotes[i][j] - allnotes[i][j-1] + corfactor
            cur_note = chr(allnotes[i][j]+65)
            if prev_note != cur_note:
                countNotesDif.append(dif)
                if dif == 89:
                    print([prev_note, cur_note, dif])
                #print(chr(allnotes[i][j]+65), end="")
                outf.write(f"{chr(dif)}")
            prev_note = cur_note
            jj+=1
        outf.write("\n")

mycntdifs = Counter(countNotesDif)


nrandomsongs = 200
randomlength=2000
with open("randomDifsongs.fa", "w") as outf:
    for i in range(nrandomsongs):
        outf.write(f">random{i}\n")
        seq = random.choices([chr(k) for k in mycntdifs.keys()], weights=list(mycntdifs.values()), k=randomlength)
        outf.write(''.join(seq))
        outf.write("\n")
                   
