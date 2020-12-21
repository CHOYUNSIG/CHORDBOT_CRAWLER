import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pygame

WIDTH = 720
HEIGHT = 480

pygame.init()
pygame.display.set_caption("chord input")

screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font50 = pygame.font.Font('com/font/OpenSans-Light.ttf', 50)
font10 = pygame.font.Font('com/font/OpenSans-Light.ttf', 10)


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

CHANNELS = 2
CHUNK = 2**10
RATE = 44100

max_head = 1
fft_data = None

freqx = np.fft.fftfreq(n = RATE)*RATE
mask = freqx > 0
for i in range(20) : mask[i] = False

freqx = freqx[mask]

fig, ax = plt.subplots(2, 1, num='data')

ax[0].set_title("audio signal")
ax[0].grid(True)
ax[0].set_xticks([])
ax[0].set_yticks([0])
ax[0].set_xlim((0, CHUNK))
audio_line, = ax[0].plot([], [], c='b', lw=0.3)

ax[1].set_title("FFT")
ax[1].grid(True)
ax[1].set_xscale("symlog")
ax[1].set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, \
    100, 200, 300, 400, 500, 600, 700, 800, 900, \
    1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, \
    10000, 15000, 20000])
ax[1].set_yticks(range(-48, 1, 6))
ax[1].set_xlim((20, RATE/2))
ax[1].set_ylim((-48, 0))
fft_line, = ax[1].plot([], [], c='g', lw=0.3)

plt.tight_layout()

def audio_init():
    audio_line.set_data([], [])
    return audio_line,

def fft_init():
    fft_line.set_data([], [])
    return fft_line,


frames, total_frames = None, None

def load_data():
    global frames, total_frames, max_head

    frames = [np.array([], dtype=np.int16) for i in range(CHANNELS)]
    total_frames = np.array([0]*CHUNK, dtype=np.int16)

    data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
    for i in range(CHUNK*CHANNELS):
        frames[i%CHANNELS] = np.append(frames[i%CHANNELS], np.array([data[i]]))

    for i in range(CHANNELS): total_frames = total_frames + frames[i]
    total_frames = total_frames / 3

    max_head = int(max(np.max(abs(total_frames)), max_head))


key = ['z','s','x','d','c', \
    'v','g','b','h','n','j','m', \
        ',','l','.',';','/', None]
chord = ['C','C#','D','Eb','E', \
    'F','F#','G','Ab','A','Bb','B', \
        'Cm','C#m','Dm','Ebm','Em', \
    'Fm','F#m','Gm','Abm','Am','Bbm','Bm']
chord_img = [font50.render(i, True, WHITE) for i in chord]
captuerd_img = font50.render('captured', True, WHITE)
capture = False

def pg_input():
    global capture
    screen.fill(BLACK)

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
                    capture = True
                break
            elif key[i+3] in pressed:
                screen.blit(chord_img[i%12+12], (100, 100))
                if 'space' in pressed:
                    screen.blit(captuerd_img, (200, 130))
                    capture = True
                break
    
    pygame.display.flip()


def audio_animate(i):
    load_data()
    pg_input()
    audio_line.set_data(np.linspace(1, CHUNK, CHUNK), total_frames)
    ax[0].set_ylim((-max_head, max_head))
    return audio_line,

def fft_animate(i):
    y = abs(np.fft.fft(total_frames, n = RATE))
    y = (np.log10(y/2**9/(max_head))*10)[mask]
    global fft_data
    fft_data = y
    fft_line.set_data(freqx, y)
    return fft_line,


p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, \
                channels=CHANNELS, \
                rate=RATE, \
                input=True, \
                frames_per_buffer=CHUNK) 

animation.FuncAnimation(fig, audio_animate, init_func=audio_init, frames=200, interval=10, blit=True)
animation.FuncAnimation(fig, fft_animate, init_func=fft_init, frames=200, interval=10, blit=True)

plt.show()
quit()
