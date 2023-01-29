import json
import math
import os
import librosa
import numpy as np
from tqdm import tqdm
from Utils import FeatureUtils as featu
from Utils import ResizeUtils as ru, FeatureUtils as fu
import audioread.ffdec as ffdec
import re

def save_specs(audio_root_dir_path, destination_dir, duration_per_track, label_subdirs=False, one_file=True, npz=False,
               sample_rate=22050, window_size=None, hop_length=None, num_segments=1, log=True, spec_shape=None,
               loop=True):
    """
    This function converts audio files to numpy arrays and stores them as either .npy or .npz
    The function expects the relevant audio files to be in subdirs of the audio_dir_path.
    The individual subdirs should only contain data for one specific class.
    The subdirs name is used as class identifier/label name.
    As standard, all data is stored into two .npy files. One for features and one for labels.

    :param loop: If set to True, loops the signal so it matches the specified size
    :param spec_shape: Used if a specific output spectrogram shape is needed for example to use with VGG16
    :param audio_root_dir_path: Root dir of all the files
    :param destination_dir: Path to save the converted files
    :param duration_per_track: The desired duration of the tracks, so they get the same size. The function will perform
    the needed padding/truncation. Only used if spec_shape is not specified.
    :param label_subdirs: If true, will make subdirs for all labels. Requires that audio filenames contains label names. Does not work if npz or one_file is set to True.
    :param one_file: If true, stores all spectrograms in one single array and saves it to disc. If false: stores one npy for every spectrogram.
    :param npz: If you want all the npy's to be stored in a single npz zip folder
    :param sample_rate: How many samples per second (high value = high memory use and vice versa)
    :param window_size: Sample size of the windows used in STFT.
    :param hop_length: How many samples the window is shifted each iteration.
    :param num_segments: How many segments each file should be divided into (keep at 1 if duration_per_track is low)
    :return:
    """

    if spec_shape is not None:
        duration_samples, window_size, hop_length = featu.specific_spec_shape(spec_shape[0], spec_shape[1])
        samples_per_track = duration_samples
        samples_per_segment = int(samples_per_track / num_segments)
        expected_windows_per_segment = spec_shape[1]
    elif window_size is not None and hop_length is not None:
        samples_per_track = sample_rate * duration_per_track
        samples_per_segment = int(samples_per_track / num_segments)
        expected_windows_per_segment = math.ceil(samples_per_segment / hop_length)
    else:
        raise ValueError("Error: Need to specify either spec_shape or window_size and hop_length")

    if one_file and npz:
        print("one_file condition overrules npz condition, and the files are saved to one file as .npy")

    b = False
    example_shape = None

    if npz:
        npz_dict = dict()

    if one_file:
        spectrograms = []

    labels = []
    mapping = dict()
    mapping_count = 0

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(audio_root_dir_path)):

        print(f"Dirpath:  {dirpath}")
        print(f"filenames: {len(filenames)}")

        if dirpath is not audio_root_dir_path:  # Skip the root folder, since we dont expect relevent files here. We expect them in the subfolders.
            label = os.path.split(dirpath)[-1]  # only works if the subfolders are equal to label names
            print(f"Saving as: \n Label: {label} \n Label num: {i}")
            print(f"Processing: {dirpath}")
            for file_name in tqdm(filenames):  # tqdm makes prints a nice loading bar in the console
                file_path = f"{dirpath}/{file_name}"
                audio_format = file_name.split(".")[-1]
                file_stem = file_name.split(".")[0]
                file_id = file_stem.split(" ")[-1]
                if audio_format == "mp3":
                    aro = ffdec.FFmpegAudioFile(file_path)
                    signal, sr = librosa.load(aro, sr=sample_rate)

                elif audio_format == "wav":
                    signal, sr = librosa.load(file_path, sr=sample_rate)

                else:
                    raise BaseException("Audio format not supported. Use wav or mp3")

                if loop:
                    padded_signal = ru.loop_or_cut(signal=signal, duration_samples=samples_per_track, sr=sr)

                else:
                    padded_signal = ru.pad_or_cut(signal, duration_samples=samples_per_track, sr=sr)

                # Divide the signals into segments, if these segments will make sense to human hearing, and have relevant sound in them (not just noise)!
                for s in range(num_segments):
                    start_sample = samples_per_segment * s
                    finish_sample = start_sample + samples_per_segment

                    stft = librosa.core.stft(padded_signal[start_sample:finish_sample],
                                             hop_length=hop_length,
                                             n_fft=window_size)

                    spectrogram = np.abs(stft)  # to get from complex numbers to absolute

                    if log:
                        spectrogram = librosa.amplitude_to_db(spectrogram)

                    rgb = np.repeat(spectrogram[..., np.newaxis], 3, -1)

                    if spec_shape is not None and spectrogram.shape != spec_shape:
                        raise ValueError(
                            f"Bad shape of {spectrogram.shape} from file {file_id}. Expected shape: {spec_shape}")

                    # first segment will represent a target shape that all segments should have, if they don't throw an exception.
                    if not b:
                        example_shape = spectrogram.shape
                        b = True

                    if spectrogram.shape != example_shape:  # Test if the shape is consistent
                        raise Exception(
                            f"Shape inconsistency error: File {file_id} has wrong shape of {spectrogram.shape}. Shape should be {example_shape}")

                    if spectrogram.shape[1] != expected_windows_per_segment:
                        raise Exception(
                            f"Shape inconsistency error: File {file_id} has wrong spectrogram shape of {spectrogram.shape[1]}. Should be {expected_windows_per_segment}")

                    if label not in mapping.keys():
                        mapping[label] = mapping_count
                        mapping_count = mapping_count + 1

                    labels.append(mapping[label])

                    segment_file_name = file_stem + "_" + str(s) + ".npy"

                    if label_subdirs:
                        if one_file:
                            spectrograms.append(spectrogram)
                        elif npz:
                            npz_dict[file_stem + "_" + str(s)] = spectrogram
                        else:
                            sub_dir = os.path.join(destination_dir, label)
                            destination_path = os.path.join(sub_dir, segment_file_name)
                            if os.path.exists(sub_dir):
                                np.save(destination_path, spectrogram)
                            else:
                                os.mkdir(sub_dir)
                                np.save(destination_path, spectrogram)
                    else:
                        if one_file:
                            spectrograms.append(spectrogram)
                        elif npz:
                            npz_dict[file_stem + "_" + str(s)] = spectrogram
                        else:
                            np.save(os.path.join(destination_dir, segment_file_name), spectrogram)

                if os.path.exists(
                        os.path.join(destination_dir, label, label)):  # To prevent a weird error that happened before
                    raise BaseException("Too many nested dirs")

    print("saving files...")
    if one_file:
        np.save(os.path.join(destination_dir, "spectrograms.npy"), np.array(spectrograms))

    elif npz:
        np.savez(os.path.join(destination_dir, "spectrograms.npz"), **npz_dict)

    np.save(os.path.join(destination_dir, "labels.npy"), np.array(labels))
    with open(os.path.join(destination_dir, "mapping.json"), "w") as fp:
        json.dump(mapping, fp, indent=4)


def save_single_spec(audio_path, destination_dir, duration, sample_rate=22050, window_size=4096, hop_length=2048,
                     num_segments=1, log=True):
    """
    Used for testing purposes

    :param audio_path: Root dir of all the files
    :param destination_path: Path to save the converted files
    :param duration: The desired duration of the tracks, so they get the same size. The function will perform the needed padding/truncation.
    :param sample_rate: How many samples per second (high value = high memory use and vice versa)
    :param window_size: Sample size of the windows used in STFT.
    :param hop_length: How many samples the window is shifted each iteration.
    :param num_segments: How many segments each file should be divided into (keep at 1 if duration_per_track is low)
    :return:
    """

    b = False
    example_shape = None

    samples_per_track = sample_rate * duration
    samples_per_segment = int(samples_per_track / num_segments)
    expected_windows_per_segment = math.ceil(samples_per_segment / hop_length)

    file_name = os.path.split(audio_path)[-1]
    label = re.split("[0-9]", file_name)[0].strip()
    audio_format = file_name.split(".")[-1]
    file_stem = file_name.split(".")[0]
    file_id = file_stem.split(" ")[-1]
    if audio_format == "mp3":
        aro = ffdec.FFmpegAudioFile(audio_path)
        signal, sr = librosa.load(aro, sr=sample_rate)

    elif audio_format == "wav":
        signal, sr = librosa.load(audio_path, sr=sample_rate)

    else:
        raise BaseException("Audio format not supported. Use wav or mp3")

    padded_signal = ru.pad_or_cut(signal, duration_seconds=duration, sr=sr)

    # Divide the signals into segments, if these segments will make sense to human hearing, and have relevant sound in them (not just noise)!
    for s in range(num_segments):
        start_sample = samples_per_segment * s
        finish_sample = start_sample + samples_per_segment

        stft = librosa.core.stft(padded_signal[start_sample:finish_sample],
                                 hop_length=hop_length,
                                 n_fft=window_size)

        spectrogram = np.abs(stft)  # to get from complex numbers to absolute

        if log:
            spectrogram = librosa.amplitude_to_db(spectrogram)

        # first segment will represent a target shape that all segments should have, if they don't throw an exception.
        if not b:
            example_shape = spectrogram.shape
            b = True

        if spectrogram.shape != example_shape:  # Test if the shape is consistent
            raise Exception(
                f"Shape inconsistency error: File {file_name} has wrong shape of {spectrogram.shape}. Shape should be {example_shape}")

        if spectrogram.shape[1] != expected_windows_per_segment:
            raise Exception(
                f"Shape inconsistency error: File {file_name} has wrong spectrogram shape of {spectrogram.shape[1]}. Should be {expected_windows_per_segment}")

        #  Måske skal segments samles i en undermappe per fil?

        segment_file_name = file_stem + "_" + str(s) + ".npy"

        destination_path = os.path.join(destination_dir, segment_file_name)

        if os.path.exists(
                os.path.join(destination_path, label, label)):  # To prevent a weird error that happened before
            raise BaseException("Too many nested dirs")

        print("saving files...")
        np.save(os.path.join(destination_path, "spectrogram.npy"), np.array(spectrogram))


def to_spec_array(audio_path, duration, sample_rate=22050, window_size=4096, hop_length=2048, num_segments=1, log=True):
    """
    Used for testing purposes

    :param audio_path: Root dir of all the files
    :param destination_path: Path to save the converted files
    :param duration: The desired duration of the tracks, so they get the same size. The function will perform the needed padding/truncation.
    :param sample_rate: How many samples per second (high value = high memory use and vice versa)
    :param window_size: Sample size of the windows used in STFT.
    :param hop_length: How many samples the window is shifted each iteration.
    :param num_segments: How many segments each file should be divided into (keep at 1 if duration_per_track is low)
    :return:
    """

    b = False
    example_shape = None

    samples_per_track = sample_rate * duration
    samples_per_segment = int(samples_per_track / num_segments)
    expected_windows_per_segment = math.ceil(samples_per_segment / hop_length)

    file_name = os.path.split(audio_path)[-1]
    audio_format = file_name.split(".")[-1]
    if audio_format == "mp3":
        aro = ffdec.FFmpegAudioFile(audio_path)
        signal, sr = librosa.load(aro, sr=sample_rate)

    elif audio_format == "wav":
        signal, sr = librosa.load(audio_path, sr=sample_rate)

    else:
        raise BaseException("Audio format not supported. Use wav or mp3")

    padded_signal = ru.pad_or_cut(signal, duration_seconds=duration, sr=sr)

    # Divide the signals into segments, if these segments will make sense to human hearing, and have relevant sound in them (not just noise)!
    for s in range(num_segments):
        start_sample = samples_per_segment * s
        finish_sample = start_sample + samples_per_segment

        stft = librosa.core.stft(padded_signal[start_sample:finish_sample],
                                 hop_length=hop_length,
                                 n_fft=window_size)

        spectrogram = np.abs(stft)  # to get from complex numbers to absolute

        if log:
            spectrogram = librosa.amplitude_to_db(spectrogram)

        # first segment will represent a target shape that all segments should have, if they don't throw an exception.
        if not b:
            example_shape = spectrogram.shape
            b = True

        if spectrogram.shape != example_shape:  # Test if the shape is consistent
            raise Exception(
                f"Shape inconsistency error: File {file_name} has wrong shape of {spectrogram.shape}. Shape should be {example_shape}")

        if spectrogram.shape[1] != expected_windows_per_segment:
            raise Exception(
                f"Shape inconsistency error: File {file_name} has wrong spectrogram shape of {spectrogram.shape[1]}. Should be {expected_windows_per_segment}")

        return spectrogram
