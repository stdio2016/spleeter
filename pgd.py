import os
import argparse

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
from tensorflow.contrib.signal import stft, hann_window, inverse_stft
from spleeter.separator import Separator
from spleeter.audio.adapter import get_default_audio_adapter
from spleeter.utils.tensor import pad_and_partition

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='input audio filename')
parser.add_argument('-o', '--output', required=True, help='output audio filename')
parser.add_argument('-p', '--params_filename', default='spleeter:4stems', help='model to attack')
parser.add_argument('--rms', type=float, default=0.01, help='attack strength')
parser.add_argument('--step', type=float, default=1, help='attack step size')
args = parser.parse_args()

filename = args.input
outputname = args.output
if '/' not in outputname:
    outputname = './' + outputname
modelname = args.params_filename

# load model
print("load model")
separator = Separator(modelname, stft_backend='tensorflow')
separator._params['attack']=4
predictor = separator._get_predictor()

# load audio
print("load audio")
audio_loader = get_default_audio_adapter()
sample_rate = 44100
waveform, _ = audio_loader.load(filename, sample_rate=sample_rate)
print(waveform.dtype)
print("max amplitude: {}".format(np.max(np.abs(waveform))))

# compute spectrogram
print("compute stft")
frame_length = separator._params['frame_length']
frame_step = separator._params['frame_step']

with predictor.graph.as_default():
    stft_feature = tf.transpose(
        stft(
            tf.transpose(waveform),
            frame_length,
            frame_step,
            window_fn=lambda frame_length, dtype: (
                hann_window(frame_length, periodic=True, dtype=dtype)),
            pad_end=True),
        perm=[1, 2, 0])

    T = separator._params['T']
    F = separator._params['F']
    spectrogram = tf.abs(pad_and_partition(stft_feature, T))[:, :, :F, :]

    stft_np = predictor.session.run(stft_feature)
    spectrogram_np = predictor.session.run(spectrogram)
    print("yes stft")

# compute perturbation
with predictor.graph.as_default():
    print("build graph")
    instrnames = []
    for iname in predictor.fetch_tensors:
        if iname.endswith('spectrogram'):
            instrnames.append(iname)
    print(instrnames)
    adv_label = instrnames.index('vocals_spectrogram')
    adv_label = np.full(spectrogram_np.shape, adv_label)
    #adv_label = np.random.randint(0, 4, size=spectrogram_np.shape)

    mixed_tensor = predictor.feed_tensors['mix_spectrogram']
    wrong_tensor = tf.placeholder(tf.float32, shape=spectrogram_np.shape)
    grad_tensors = []
    for i, iname in enumerate(instrnames):
        part_tensor = predictor.fetch_tensors[iname]
        g = tf.gradients(part_tensor, mixed_tensor, grad_ys=wrong_tensor)[0]
        grad_tensors.append(g)

    m = 0
    M0 = 200
    gamma = args.step
    perturb_allowance = args.rms
    X0 = spectrogram_np
    X = spectrogram_np.copy()
    early_exit = 5
    best_so_far = 1.0
    best_m = 0
    best_X = X0
    r = np.zeros(X.shape)

    n_frame = X.shape[0]*X.shape[1]
    n_bin = X.shape[2]
    n_ch = X.shape[3]
    frame_per_second = int(np.ceil(sample_rate / frame_step))
    secs = int(np.ceil(n_frame / frame_per_second))
    n_pad = secs * frame_per_second - n_frame
    X_t = X.reshape((n_frame, n_bin*n_ch))
    X_t = np.concatenate([X_t, np.zeros((n_pad, n_bin*n_ch), dtype=np.float32)])
    X_t = X_t.reshape((secs, frame_per_second * n_bin*n_ch))

    # this number doesn't affect result
    n_sq = np.sqrt(n_bin*n_ch)

    avg_mag = np.linalg.norm(X_t, axis=1)
    print("average magnitude:", np.average(avg_mag/n_sq))
    print(avg_mag.shape)
    print("max magnitude:", np.max(X))
    while m < M0:
        print("compute prediction")
        prediction = predictor({
            'audio_id': '',
            'mix_spectrogram': X})
        score = [prediction[iname] for iname in instrnames]
        score = np.array(score)
        if m == 0:
            orig_label = np.argmax(score, axis=0)
        pred_label = np.argmax(score, axis=0)
        wrong = (pred_label != adv_label).astype(np.float32)
        wrong = np.average(wrong)
        print("wrong: {}".format(wrong))
        if wrong < best_so_far:
            best_m = m
            best_so_far = wrong
            best_X = X.copy()
        elif m >= best_m + early_exit:
            print("error rate does not improve in %d steps, early exit" % early_exit)
            X = best_X
            break
        print("compute gradient")
        r_m = np.zeros(X.shape, dtype=np.float32)
        for i, iname in enumerate(instrnames):
            #wrong = score[i] * ((i == adv_label).astype(np.float32) * 2 - 1)
            wrong = (orig_label == pred_label).astype(np.float32)
            wrong *= (i == adv_label).astype(np.float32) - (i == orig_label)
            grad = grad_tensors[i].eval({
                mixed_tensor: X,
                wrong_tensor: wrong
            }, session=predictor.session)
            r_m += grad
        r_m *= gamma / np.max(np.abs(r_m))
        r = r + r_m
        # reshape as each second
        r_part = r.reshape((n_frame, n_bin*n_ch))
        r_part = np.concatenate([r_part, np.zeros((n_pad, n_bin*n_ch), dtype=np.float32)])
        r_part = r_part.reshape((secs, frame_per_second*n_bin*n_ch))
        # compute volume of each second
        r_amp = np.linalg.norm(r_part, axis=1)
        # project to perturbation bound
        r_part *= np.clip(avg_mag * perturb_allowance / (r_amp + 1e-7), -1, 1).reshape((-1,1))
        r = r_part.reshape((-1, n_bin, n_ch))
        r = r[:-n_pad]
        r = r.reshape((X.shape[0], X.shape[1], n_bin, n_ch))

        X_next = X0 + r

        diff = X_next - X0
        diff = diff.reshape((n_frame, n_bin*n_ch))
        diff = np.concatenate([diff, np.zeros((n_pad, n_bin*n_ch), dtype=np.float32)])
        diff = diff.reshape((secs, frame_per_second * n_bin*n_ch))
        perturb_mag = np.linalg.norm(diff, axis=1) / perturb_allowance
        print("perturbation L2:", np.average(perturb_mag/n_sq))
        print("perturbation Linf:", np.max(np.abs(X_next - X0)))
        print("perturbation ratio:", np.max(perturb_mag / (avg_mag + 1e-7)))
        X = X_next
        m += 1

#prediction = separator.separate(X)
#print(prediction)

#for stem in prediction:
#    part = prediction[stem]
#    part = part.flatten().reshape((-1,2))
#    part *= 1/512
#    audio_loader.save(outfolder + '/' + stem + '.wav', part, sample_rate=sample_rate)

# convert spectrogram back to waveform

EPSILON = 1e-10
WINDOW_COMPENSATION_FACTOR = 2./3.

# extend spectrogram
n_extra_row = frame_length // 2 + 1 - F
extended = np.zeros([X.shape[0], X.shape[1], n_extra_row, X.shape[-1]], dtype=np.float32)
X_ext = np.concatenate([X, extended], axis=2)

# create mask of spectrogram
old_shape = X_ext.shape
new_shape = [old_shape[0]*old_shape[1], old_shape[2], old_shape[3]]
X_mask = X_ext.reshape(new_shape)
X_mask = X_mask[:stft_feature.shape[0], ...]
X_mask = X_mask / (np.abs(stft_np) + EPSILON)
new_stft = X_mask * stft_np
new_stft[:,-n_extra_row:] = stft_np[:,-n_extra_row:]
print(new_stft.shape, stft_np.shape)

# inverse transform
with predictor.graph.as_default():
    print("inverse fft start")
    stft_ph = tf.placeholder(tf.complex64, new_stft.shape)
    inversed = inverse_stft(
        tf.transpose(stft_ph, perm=[2, 0, 1]),
        frame_length,
        frame_step,
        window_fn=lambda frame_length, dtype: (
            hann_window(frame_length, periodic=True, dtype=dtype))
    ) * WINDOW_COMPENSATION_FACTOR
    reshaped = tf.transpose(inversed)
    time_crop = waveform.shape[0]
    new_waveform = reshaped[:time_crop, :]
    print("inverse fft")
    new_waveform = predictor.session.run(new_waveform, {stft_ph: new_stft})
    print("inferse fft finished")

print("perturnation: {}".format(np.max(np.abs(new_waveform - waveform))))
d = new_waveform - waveform
d *= d
print("perturnation RMS: {}".format(np.sqrt(np.average(d))))

audio_loader.save(outputname, new_waveform, sample_rate=sample_rate)
