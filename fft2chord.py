
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

CHUNK = 2**10
RATE = 44100

WIDTH = 800
HEIGHT = 600
FPS = 60

THR_ATK_INIT = 10
THR_ATK = 5

import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import matplotlib.animation as animation


import pygame
pygame.init()
pygame.display.set_caption("chord input")

screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font50 = pygame.font.Font('com/font/OpenSans-Light.ttf', 50)
font10 = pygame.font.Font('com/font/OpenSans-Light.ttf', 10)


import pyaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, \
                channels=1, \
                rate=RATE, \
                input=True, \
                frames_per_buffer=CHUNK) 


import numpy as np

from scipy.fftpack import *

import time


max_head = 1
audio_data = None
fft_data = None


smplx = np.linspace(1, CHUNK, CHUNK)

freqx = fftfreq(n = RATE)*RATE
mask = freqx > 0
for i in range(20) : mask[i] = False
freqx = freqx[mask]

####plot generation start
fig, ax = plt.subplots(2, 1, num='data')
fig.set_figwidth(WIDTH/fig.dpi)
fig.set_figheight(HEIGHT/fig.dpi)

ax[0].set_title("audio signal")
ax[0].grid(True)
ax[0].set_xticks([])
ax[0].set_xlim((0, CHUNK))
ax[0].set_ylim((-1, 1))
ax[0].set_yticks([-1, -0.5, 0, 0.5, 1])
audio_line, = ax[0].plot([], [], c='b', lw=0.5)

ax[1].set_title("FFT")
ax[1].grid(True)
ax[1].set_xscale("symlog")
ax[1].set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, \
    100, 200, 300, 400, 500, 600, 700, 800, 900, \
    1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, \
    10000, 15000, 20000])
ax[1].set_yticks(range(-96, 1, 12))
ax[1].set_xlim((20, RATE/2))
ax[1].set_ylim((-96, 0))
fft_line, = ax[1].plot([], [], c='g', lw=0.5)

plt.tight_layout()
####plot generation end

thr_time = time.time()
thr_accel = None

def load_data():
    global audio_data, fft_data, max_head, thr_time, thr_accel
    audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
    fft_data = np.log10(abs(fft(audio_data, n = RATE))/(max_head * CHUNK)*2)[mask]*20
    
    temp_head = np.max(abs(audio_data))
    if max_head <= temp_head:
        thr_accel = max_head = temp_head
        thr_time = time.time()
    else:
        interval_time = time.time() - thr_time - THR_ATK_INIT
        if 0 < interval_time < THR_ATK:
            max_head = max(int(thr_accel * (1 - interval_time / THR_ATK)), temp_head)

    ax[0].set_ylim((-max_head, max_head))

def audio_init():
    audio_line.set_data([], [])
    return audio_line, 

def fft_init():
    fft_line.set_data([], [])
    return fft_line,

def audio_animate(i):
    load_data()
    audio_line.set_data(smplx, audio_data)
    return audio_line,

def fft_animate(i):
    fft_line.set_data(freqx, fft_data)
    return fft_line,

key = ['z','s','x','d','c', \
    'v','g','b','h','n','j','m', \
        ',','l','.',';','/', None]
chord = ['C','C#','D','Eb','E', \
    'F','F#','G','Ab','A','Bb','B', \
        'Cm','C#m','Dm','Ebm','Em', \
    'Fm','F#m','Gm','Abm','Am','Bbm','Bm']
chord_img = [font50.render(i, True, BLACK) for i in chord]
captuerd_img = font50.render('captured', True, BLACK)


animation.FuncAnimation(fig, frames=200, interval=10, blit=True, func=audio_animate, init_func=audio_init)
animation.FuncAnimation(fig, frames=200, interval=10, blit=True, func=fft_animate, init_func=fft_init)
plt.show()

'''
while True:
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit()
    
    pressed = [pygame.key.name(k)for k,v in enumerate(pygame.key.get_pressed()) if v]
    for i in range(14):
        if key[i] in pressed:
            if key[i+4] in pressed:
                screen.blit(chord_img[i%12], (100, 100))
                if 'space' in pressed:
                    screen.blit(captuerd_img, (200, 130))
                break
            elif key[i+3] in pressed:
                screen.blit(chord_img[i%12+12], (100, 100))
                if 'space' in pressed:
                    screen.blit(captuerd_img, (200, 130))
                break
    
    pygame.display.flip()

quit()
'''