
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

CHUNK = 2**10
RATE = 44100

WIDTH_PG = 640
HEIGHT_PG = 480
WIDTH_MP = 800
HEIGHT_MP = 600

FPS = 60

THR_ATK_INIT = 10
THR_ATK = 5

import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import matplotlib.animation as animation


import pygame
pygame.init()
pygame.display.set_caption("chord input")

screen = pygame.display.set_mode((WIDTH_PG, HEIGHT_PG))
clock = pygame.time.Clock()
font50 = pygame.font.Font('com/font/OpenSans-Light.ttf', 50)
font20 = pygame.font.Font('com/font/OpenSans-Light.ttf', 20)


import pyaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, \
                channels=1, \
                rate=RATE, \
                input=True, \
                frames_per_buffer=CHUNK) 


import numpy as np
from scipy.fftpack import *
import struct
import csv
import time


max_head = 1
audio_data = None
fft_data = None

delay = 0
with open("com/delaytick.dat", 'rb') as f:
    delay, = struct.unpack('d', f.read())

smplx = np.linspace(1, CHUNK, CHUNK)

freqx = fftfreq(n = RATE)*RATE
mask = freqx > 0
for i in range(20) : mask[i] = False
freqx = freqx[mask]

####plot generation start
fig, ax = plt.subplots(2, 1, num='data')
fig.set_figwidth(WIDTH_MP/fig.dpi)
fig.set_figheight(HEIGHT_MP/fig.dpi)

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

red_dot = []

class fft_capture():
    def __init__(self, chord, test):
        self.init_time = time.time()
        self.chord = chord
        self.test = test
        self.end = False
    
    def process(self):
        if time.time() - self.init_time >= delay:
            self.end = True
            if self.test:
                pass
            else:
                with open("data/" + self.chord + ".csv", 'a', encoding='utf-8', newline='') as f:
                    csv.writer(f).writerow(fft_data)

    def __del__(self):
        global red_dot
        red_dot.append(time.time())

fft_captures = []

def capture_manager():
    for fftcpt in fft_captures:
        fftcpt.process()
        if fftcpt.end:
            fftcpt.__del__()
            fft_captures.remove(fftcpt)


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


key = ['z','s','x','d','c', \
    'v','g','b','h','n','j','m', \
        ',','l','.',';','/', None]
chord = ['C','C#','D','Eb','E', \
    'F','F#','G','Ab','A','Bb','B', \
        'Cm','C#m','Dm','Ebm','Em', \
    'Fm','F#m','Gm','Abm','Am','Bbm','Bm']

img_chord = [font50.render(i, True, WHITE) for i in chord]
img_captured = font20.render('captured', True, WHITE)
img_movemode = font20.render('Window move mode (Q)', True, WHITE)
img_dsetmode = font20.render('Delay set mode (W)', True, WHITE)

pre_pressed = []
move_mode = True
dset_mode = False

def pg_input():
    global fft_captures, pre_pressed, move_mode, dset_mode, red_dot, delay
    clock.tick(FPS)
    current = time.time()
    screen.fill(BLACK)

    if move_mode:
        screen.blit(img_movemode, (10, HEIGHT_PG-50))
    else: 
        for event in pygame.event.get():
            pass
    
    pressed = [pygame.key.name(k) for k,v in enumerate(pygame.key.get_pressed()) if v]
    
    if 'q' in pressed and 'q' not in pre_pressed:
        move_mode = not move_mode
    
    if 'escape' in pressed:
        quit()

    if move_mode:
        pass
    else:
        if 'w' in pressed and 'w' not in pre_pressed:
            dset_mode = not dset_mode
        
        if dset_mode:
            screen.blit(img_dsetmode, (10, HEIGHT_PG-50))
            if 'left' in pressed and 'right' not in pressed:
                delay -= 0.01
                screen.blit(font20.render('delay : %.2f'%delay, True, WHITE), (10, HEIGHT_PG-30))    
                with open("com/delaytick.dat", "wb") as f:
                    f.write(struct.pack("d", delay))
            if 'right' in pressed and 'left' not in pressed:
                delay += 0.01
                screen.blit(font20.render('delay : %.2f'%delay, True, WHITE), (10, HEIGHT_PG-30))    
                with open("com/delaytick.dat", "wb") as f:
                    f.write(struct.pack("d", delay))

        for i in range(14):
            if key[i] in pressed:
                if key[i+4] in pressed:
                    screen.blit(img_chord[i%12], (70, 30))
                    if 'space' in pressed:
                        screen.blit(img_captured, (180, 70))
                        fft_captures.append(fft_capture(chord[i%12], dset_mode))
                    break
                elif key[i+3] in pressed:
                    screen.blit(img_chord[i%12+12], (70, 30))
                    if 'space' in pressed:
                        screen.blit(img_captured, (180, 70))
                        fft_captures.append(fft_capture(chord[i%12+12], dset_mode))
                    break
    
    for i in red_dot:
        if current - i >= 0.1:
            red_dot.remove(i)
    
    if len(red_dot) > 0:
        pygame.draw.circle(screen, RED, (WIDTH_PG-20, 20), 10)
    
    pre_pressed = pressed
    pygame.display.flip()


def audio_init():
    audio_line.set_data([], [])
    return audio_line, 

def fft_init():
    fft_line.set_data([], [])
    return fft_line,

def audio_animate(i):
    load_data()
    capture_manager()
    audio_line.set_data(smplx, audio_data)
    return audio_line,

def fft_animate(i):
    pg_input()
    fft_line.set_data(freqx, fft_data)
    return fft_line,


animation.FuncAnimation(fig, frames=200, interval=10, blit=True, func=audio_animate, init_func=audio_init)
animation.FuncAnimation(fig, frames=200, interval=10, blit=True, func=fft_animate, init_func=fft_init)
plt.show()
