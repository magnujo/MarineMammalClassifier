import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from sympy import symbols, solve



#  Returns the samples and sampling rate from the audio file
def get_wave(audio_path, sample_rate=22050, display=False):
    signal, sr = librosa.load(audio_path, sr=sample_rate)
    if display:
        librosa.display.waveshow(signal, sr=sr)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()
    return signal, sr


def specific_spec_shape(num_freq_bins, num_time_bins):
    """
    Solves the equations from the Digital Sound section of the thesis paper to find the parameters needed
    to obtain the specified values, which can be used to produce spectrograms of a specific shape.
    :param num_freq_bins:
    :param num_time_bins:
    :return:
    """

    # Solve for N_m (frame size) for a specific number of frequency bins
    N_m = symbols('N_m')
    N_m = int(solve(((N_m / 2) + 1) - num_freq_bins, N_m)[0])

    # Solve for N (signal size)
    N = symbols("N")
    H = int((N_m / 2))
    z = (((N - N_m) / H) + 3) - num_time_bins
    N = int(solve(z, N)[0])

    return N, N_m, H


def get_melspectrogram_from_path(file_path, sample_rate=22050, display=False, n_mels=64, window_size=1024, hop_length=512, log=True):
    signal, sr = librosa.load(file_path, sr=sample_rate)
    melspec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels, n_fft=window_size, hop_length=hop_length)
    if log:
        melspec = librosa.amplitude_to_db(melspec)  # apply log to amplitude to get decibel
    if display:
        librosa.display.specshow(melspec, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.f")
        plt.show()
    return melspec

def get_spectrogram_from_path(file_path, sample_rate=22050, display=False, window_size=2048, hop_length=512, log=True):
    signal, sr = librosa.load(file_path, sr=sample_rate)
    window_size = window_size  # 2048 samples per window
    hop_length = hop_length  # how many samples we are moving for every fft
    stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=window_size)
    spectrogram = np.abs(stft)  # to get from complex numbers to absolute

    if log:
        spectrogram = librosa.amplitude_to_db(spectrogram)  # converts amplitude to decibel (using log)
    if display:
        librosa.display.specshow(spectrogram, sr=sample_rate, hop_length=hop_length, y_axis="log", x_axis="time")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar()
        plt.show()
    return spectrogram


def get_spectrogram_from_array(audio_signal, sample_rate=22050, display=False, window_size=2048, hop_length=512, log_amp=True):
    window_size = window_size  # 2048 samples per window
    hop_length = hop_length  # how many samples we are moving for every fft
    stft = librosa.core.stft(audio_signal, hop_length=hop_length, n_fft=window_size)
    spectrogram = np.abs(stft)  # to get from complex numbers to absolut
    if display:
        display_spectrogram(spectrogram, sample_rate, hop_length, log_amp)
    return spectrogram

def display_spectrogram(spectrogram, sample_rate, hop_length, log_amp=True, log_freq=True):
    if log_amp:
        spectrogram = librosa.amplitude_to_db(spectrogram)  # converts amplitude to decibel (using log)
    if log_freq:
        librosa.display.specshow(spectrogram, sr=sample_rate, hop_length=hop_length, y_axis="log")
    else:
        librosa.display.specshow(spectrogram, sr=sample_rate, hop_length=hop_length, y_axis="hz")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()
    plt.show()
    return spectrogram

