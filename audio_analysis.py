import librosa
import os
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd


data = {}
for file in os.listdir("./"):
    if file.endswith(".mp3"):
        print(file)
        data[file] = librosa.load(file, sr=22000)

##len(data['psarantonis1.mp3'][0])

plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
