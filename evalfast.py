import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import os
import numpy as np
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
    tracks.append(track)
    filename2 = os.path.join(args.estimate, i+'.wav')
    estimate, _ = audio_loader.load(filename2, sample_rate=sample_rate)
    estimates.append(estimate)
tracks = np.array(tracks)
estimates = np.array(estimates)
error = tracks - estimates
errorl2 = np.linalg.norm(error.reshape((4,-1)), axis=1)
l2 = np.linalg.norm(tracks.reshape((4,-1)), axis=1)
print(np.log(l2 / errorl2) / np.log(10) * 20)
