"""
The functions in this file are from https://www.tensorflow.org/tutorials/audio/simple_audio
"""

import tensorflow as tf
import numpy as np


def npy_to_dataset(spectrograms, labels):

    # Add a new dimension to the input so each input has the shape (173, 13, 1) (like a RGB image)
    spectrograms = spectrograms[..., np.newaxis]  # shape: (num_samples, num_freq_bins, num_time_bins, 1)
    ds = tf.data.Dataset.from_tensor_slices((spectrograms, labels))
    return ds



def squeeze(audio, labels):
    """
    https://www.tensorflow.org/tutorials/audio/simple_audio
    :param audio:
    :param labels:
    :return:
    """
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels



def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def convert_spectrogram(spectrogram):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def split_tf_ds(ds):
    ds.shuffle()
    train_ds = ds.shard(num_shards=2, index=0)
    val_ds = ds.shard(num_shards=2, index=1)

    test_ds = val_ds.shard(num_shards=2, index=0)
    val_ds = val_ds.shard(num_shards=2, index=1)

    return train_ds, val_ds, test_ds

def make_tf_ds(audio_root_folder, duration, sr):
    sample_len = duration * sr

    # Automatically converts a root dir to a dataset. Can split data if needed.
    ds = tf.keras.utils.audio_dataset_from_directory(
        directory=audio_root_folder,
        batch_size=None,  # Set to None?
        labels="inferred",  # Uses the sub folder names as labels
        seed=0,
        output_sequence_length=sample_len)  # cuts/pads. If None: Everything gets padded to the len of the longest wav in a batch)
    label_names = np.array(ds.class_names)

    ds = ds.map(squeeze, tf.data.AUTOTUNE)  # remove channal dimension
    ds = ds.map(map_func=lambda audio, label: (get_spectrogram(audio), label),
                num_parallel_calls=tf.data.AUTOTUNE)

    return ds

def save_tf_ds(ds, destination):
    ds.save(destination)

def make_spec_ds(ds):
    return ds.map(
        map_func=lambda audio, label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE)

