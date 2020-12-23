RATE = 44100
import matplotlib.pyplot as plt
import numpy as np
import csv

chord = input('Chord : ')


x = range(20, RATE//2, 1)
chords = ['C','C#','D','Eb','E', \
    'F','F#','G','Ab','A','Bb','B', \
        'Cm','C#m','Dm','Ebm','Em', \
    'Fm','F#m','Gm','Abm','Am','Bbm','Bm']

if chord not in chords: quit()

freq = [32.7032, 34.6478, 36.7081, 38.8909, 41.2034, \
    43.6535, 46.2493, 48.9994, 51.9130, 55.0000, 58.2705, 61.7354] * 2

xtick = np.array([])

for i in range(len(chords)):
    if chords[i] == chord:
        if i < 12:
            xtick = np.append(xtick, [freq[i], freq[i+4], freq[i+7]])
        else:
            xtick = np.append(xtick, [freq[i-12], freq[i-12+3], freq[i-12+7]])
        for j in range(7):
            xtick = np.append(xtick, xtick[3*j + 0]*2)
            xtick = np.append(xtick, xtick[3*j + 1]*2)
            xtick = np.append(xtick, xtick[3*j + 2]*2)
        break


print(xtick)
with open('data/'+chord+'.csv', 'r') as f:
    rdr = csv.reader(f)
    for fft in rdr:
        plt.figure(figsize=(10, 2), num='data')
        plt.plot(x, list(map(float, fft)), c='r', lw=0.5)
        plt.grid(True)
        plt.xscale('symlog')
        plt.xticks(xtick)
        plt.xlim((20, RATE//2))
        plt.ylim((-96, 0))
        plt.show()