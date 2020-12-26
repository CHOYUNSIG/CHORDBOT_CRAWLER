import matplotlib.pyplot as plt
import numpy as np
import csv
from constants import *

c = input('Chord : ')

x = range(20, RATE//2, 1)

if c not in CHORD: quit()
else : c = CHORD.index(c)

xtick = np.array([])
if c < 12:
    for i in range(8):
        try:
            xtick = np.append(xtick, FREQ[c + 12*i + 0])
            xtick = np.append(xtick, FREQ[c + 12*i + 4])
            xtick = np.append(xtick, FREQ[c + 12*i + 7])
        except:
            pass
else:
    for i in range(8):
        try:
            xtick = np.append(xtick, FREQ[c + 12*i + 0])
            xtick = np.append(xtick, FREQ[c + 12*i + 3])
            xtick = np.append(xtick, FREQ[c + 12*i + 7])
        except:
            pass
print(xtick)

with open('data/'+CHORD[c]+'.csv', 'r') as f:
    rdr = csv.reader(f)
    for fft in rdr:
        plt.figure(figsize=(10, 2), num=CHORD[c])
        plt.plot(x, list(map(float, fft)), c='r', lw=0.5)
        plt.grid(True)
        plt.xscale('symlog')
        plt.xticks(xtick)
        plt.xlim((20, RATE//2))
        plt.ylim((-96, 0))
        plt.show()