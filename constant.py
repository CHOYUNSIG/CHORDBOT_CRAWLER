
import numpy as np

RATE = 44100


FREQ = np.array([32.7032, 34.6478, 36.7081, 38.8909, 41.2034, \
    43.6535, 46.2493, 48.9994, 51.9130, 55.0000, 58.2705, 61.7354])
for i in range(7):
    FREQ = np.append(FREQ, FREQ[:12]*2)


CHORD = ['C','C#','D','Eb','E', \
    'F','F#','G','Ab','A','Bb','B', \
        'Cm','C#m','Dm','Ebm','Em', \
    'Fm','F#m','Gm','Abm','Am','Bbm','Bm']