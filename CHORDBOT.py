
THR_ATK_INIT = 10
THR_ATK = 5

CHUNK = 2**10
RATE = 44100

WIDTH_MP = 640
HEIGHT_MP = 480

import tensorflow as tf
import numpy as np
from scipy.fftpack import *
import time
import pyaudio

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, \
                channels=1, \
                rate=RATE, \
                input=True, \
                frames_per_buffer=CHUNK)

max_head = 1
audio_data = None
fft_data = None
chord_data = None

chord = ['C','C#','D','Eb','E', \
    'F','F#','G','Ab','A','Bb','B', \
        'Cm','C#m','Dm','Ebm','Em', \
    'Fm','F#m','Gm','Abm','Am','Bbm','Bm']

smplx = np.linspace(1, CHUNK, CHUNK)

freqx = fftfreq(n = RATE)*RATE
mask = freqx > 0
for i in range(20) : mask[i] = False
freqx = freqx[mask]

chordx = list(range(24))

####tensorflow
sess = tf.compat.v1.Session()

saver = tf.compat.v1.train.import_meta_graph('model/model.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('model/'))

graph = tf.compat.v1.get_default_graph()
x = graph.get_tensor_by_name("in:0")
y = graph.get_tensor_by_name("out:0")
####tensorflow end

thr_time = time.time()
thr_accel = None

def load_data():
    global audio_data, fft_data, chord_data, max_head, thr_time, thr_accel
    audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
    fft_data = np.round(np.log10(abs(fft(audio_data, n = RATE))/(max_head * CHUNK)*2)[mask]*20, 4)

    temp_head = np.max(abs(audio_data))
    if max_head <= temp_head:
        thr_accel = max_head = temp_head
        thr_time = time.time()
    else:
        interval_time = time.time() - thr_time - THR_ATK_INIT
        if 0 < interval_time < THR_ATK:
            max_head = max(int(thr_accel * (1 - interval_time / THR_ATK)), temp_head)

    chord_data = sess.run(y, feed_dict={x: [fft_data]})

while True:
    time.sleep(1)
    load_data()
    print(audio_data, fft_data, chord_data)