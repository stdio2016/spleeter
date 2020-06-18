import museval
import argparse
import os
import numpy as np
from museval.metrics import bss_eval
from spleeter.audio.adapter import get_default_audio_adapter

parser = argparse.ArgumentParser()
parser.add_argument('original')
parser.add_argument('estimate')
args = parser.parse_args()

print("load audio")
audio_loader = get_default_audio_adapter()
sample_rate = 44100
tracks = []
estimates = []
for i in ['bass', 'drums', 'other', 'vocals']:
    filename = os.path.join(args.original, i+'.wav')
    track, _ = audio_loader.load(filename, sample_rate=sample_rate)
    a = np.sum(track, axis=1)
    print(np.all(a == 0))
    tracks.append(track + 1e-7)
    filename2 = os.path.join(args.estimate, i+'.wav')
    estimate, _ = audio_loader.load(filename2, sample_rate=sample_rate)
    estimates.append(estimate + 1e-7)
tracks = np.array(tracks)
estimates = np.array(estimates)
print(tracks.shape)
print(estimates.shape)
ans = bss_eval(tracks, estimates)
SDR = ans[0]
print(np.average(SDR, axis=1))
