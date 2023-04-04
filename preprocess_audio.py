import glob
import os

import numpy as np
import pandas as pd
import torch
import librosa
# TODO: for each frame get the required features

frame_size = 0.05 # in seconds
target_sample_rate = 16000

audio_file_paths = glob.glob('data/audio/*')

# TODO: experiment with threshold values
is_talking_threshold = 0.6 # Experimental: Used to classify whether someone is talking or not
is_voiced_threshold = 0.1 # Experimental: Used to classify whether speech is voiced or unvoiced

voice_activity_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True, onnx=True)

def get_is_talking(frame, sample_rate):
    frame_copy = np.array(frame, copy=True)
    is_talking_prob = voice_activity_model(torch.from_numpy(frame_copy), sr=sample_rate)
    return True if is_talking_prob > is_talking_threshold else False

def get_pitch(frame, sample_rate):
    chroma = librosa.feature.chroma_stft(y=frame, sr=sample_rate, n_fft=len(frame))

    absolute_pitch = librosa.pitch_tuning(chroma)
    relative_pitch = librosa.estimate_tuning(y=frame, sr=sample_rate, n_fft=len(frame))

    return absolute_pitch, relative_pitch


def get_is_voiced(frame, is_talking):
    zcr = librosa.feature.zero_crossing_rate(frame)
    low_zcr_frames = sum(zcr[0] < is_voiced_threshold)
    frac_low_zcr_frames = low_zcr_frames / zcr.shape[1]

    return True if frac_low_zcr_frames > 0.5 and is_talking else False

def get_intensity(frame):
    power_spectogram = np.abs(librosa.stft(frame, n_fft=len(frame)))**2
    db_array = librosa.power_to_db(power_spectogram)
    return np.max(db_array)

def get_spectral_stability(frame, sample_rate):
    return None



for file_path in audio_file_paths:
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    feature_list = []

    # Load audio file, adjust and normalize sample rate, and split it into seperate frames
    orig_audio_signal, orig_sample_rate = librosa.load(file_path)
    resampled_audio_signal = librosa.resample(y=orig_audio_signal, orig_sr=orig_sample_rate, target_sr=target_sample_rate)
    normalized_audio_signal = librosa.util.normalize(resampled_audio_signal)
    frame_length = int(frame_size * target_sample_rate)
    frames = librosa.util.frame(normalized_audio_signal, frame_length=frame_length, hop_length=frame_length)


    for i, _ in enumerate(frames):
        frame = frames[:, i]
        is_talking_prob = get_is_talking(frame, sample_rate=target_sample_rate)
        absolute_pitch, relative_pitch = get_pitch(frame, sample_rate=target_sample_rate)
        is_voiced = get_is_voiced(frame, is_talking_prob)
        intensity = get_intensity(frame)
        # TODO: implement spectral stability
        # spectral_stability = get_spectral_stability(frame, sample_rate=target_sample_rate)

        features = [is_talking_prob, absolute_pitch, relative_pitch, is_voiced, intensity]
        feature_list.append(features)

    feature_df = pd.DataFrame(feature_list)
    feature_df.to_csv('./data/features/'+ file_name + '.csv', header=['is_talking', 'absolute_pitch', 'relative_pitch', 'is_voiced', 'intensity'])



# 1) Voice activity: binary feature representing current voice activity of two speakers (speech/no speech)
# Extracted from the manual annotation of the corpus
# -- 1 feature

# 2) Pitch: use Snack Toolkit to transform audio into semitones and then apply z-normalization
# Both relative and absolute values were used as individual features
# Additionally, a binary feature indicating whether the current frame was voiced or unvoiced was included
# -- 3 feature

# 3) Power: the power (intensity) in dB was extracted using Snack and then z-normalized for the individual speaker
# -- 1 feature

# 4) Spectral stability: t.b.d.
# -- 1 feature
