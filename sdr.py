import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import os
import numpy as np
from spleeter.audio.adapter import get_default_audio_adapter

parser = argparse.ArgumentParser()
parser.add_argument('original')
parser.add_argument('perturbed')
args = parser.parse_args()

print("load audio")
audio_loader = get_default_audio_adapter()
sample_rate = 44100

filename = args.original
track, _ = audio_loader.load(filename, sample_rate=sample_rate)
filename2 = args.perturbed
estimate, _ = audio_loader.load(filename2, sample_rate=sample_rate)

error = track - estimate
errorl2 = np.linalg.norm(error.flatten())
l2 = np.linalg.norm(track.flatten())
print(np.log(l2 / errorl2) / np.log(10) * 20)
