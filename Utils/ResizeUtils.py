import math
import os
import random
import librosa
import numpy as np
from scipy.signal.windows import tukey

DATA_PATH = r"C:\datasets\whalefm"
JSON_PATH = r"C:\datasets\whalefm\data.json"
TEST_FILE = r"C:\datasets\whalefm\pilot\1_pilotwhale.mp3"
SAMPLE_RATE = 22050  # samples per second
MEAN_DURATION = round(3.9881760626632525)



def mean_duration(root_audio_dir, sr=22050):
    """
    Calculates the mean duration of all the files in root audio dir
    Returns the mean in seconds. Use to find a suitable common duration for all the files.

    :param root_audio_dir: the root dir of the audio files. Can contain files and folders.
    :param sr: sample rate used to load the audio. Sample rate = number of samples per second
    :return: return the mean in seconds
    """

    total = 0
    count = 0
    for (current_dir, dir_names, filenames) in os.walk(root_audio_dir):
        print(f"Searching: {current_dir}")
        for filename in filenames:
            file = f"{current_dir}/{filename}"
            samples, _ = librosa.load(file, sr=SAMPLE_RATE)
            seconds = len(samples) / sr  # divide the number of samples by sample rate to get its duration in seconds
            total = total + seconds
            count = count + 1
            if count % 1000 == 0:
                print("1000 files read")
    mean = total / count
    print(f"Found mean: {mean} from {count} files")
    return mean


def median_duration(root_audio_dir, sr=22050):
    """
    Calculates the median duration of all the files in root audio dir
    Returns the median in seconds. Use to find a suitable common duration for all the files.

    :param root_audio_dir: the root dir of the audio files. Can contain files and folders.
    :param sr: sample rate used to load the audio. Sample rate = number of samples per second
    :return: return the median in seconds
    """

    durations = []

    count = 0
    for (current_dir, dir_names, filenames) in os.walk(root_audio_dir):
        print(f"Searching: {current_dir}")
        for filename in filenames:
            file = f"{current_dir}/{filename}"
            samples, _ = librosa.load(file, sr=SAMPLE_RATE)
            seconds = len(samples) / sr  # divide the number of samples by sample rate to get its duration in seconds
            durations.append(seconds)
            count = count + 1
            if count % 1000 == 0:
                print("1000 files read")
    durations.sort()
    median = durations[int(len(durations) / 2)]
    print(f"Found median: {median} from {count} files")
    return median


def cut(signal, goal_len):
    signal_len = len(signal)
    cut_len = signal_len - goal_len
    cut_begin = random.randint(0, cut_len)  # Number of samples to cut of, in the beginning
    cut_end = cut_len - cut_begin  # Number of samples to cut off in the end
    output = signal[cut_begin:-cut_end]  # The first cut_begin samples and the last cut_end samples gets cut off.
    return output


def loop(signal, goal_len, alpha=0.02):
    """
    Loops the signal to the specified goal_len
    :param signal:
    :param goal_len:
    :param alpha: If set to 1 there will a very slow descent towards 0 at the end
    of each loop and a very slow ascent towards 1 at the start of every loop making the transition between loops
     very gradual. If set to 0 there will be no transition.
    :return:
    """

    # Tukey window is used as its flexible with the alpha value
    tukey_ = lambda m: tukey(M=m, alpha=alpha, sym=True)

    signal_len = len(signal)
    num_loops = math.ceil(goal_len / signal_len)

    # To only make the loop transitions smooth different windows are needed for the start, end and middle part of the
    # loop.
    tukeys = {"start": np.concatenate(
        [np.ones(math.ceil(signal_len / 2)), tukey_(len(signal))[-math.floor(signal_len / 2):]]),
              "middle": tukey_(len(signal)),
              "end": np.concatenate(
                  [tukey_(len(signal))[:math.floor(signal_len / 2)], np.ones(math.ceil(signal_len / 2))])}

    # Apply the start, end and middle windows to the signal.
    start, middle, end = (tukeys["start"] * signal, tukeys["middle"] * signal, tukeys["end"] * signal)

    # If there are more then two loops we need to use the middle signal.
    if num_loops > 2:
        looped_signal = np.concatenate([start, list(middle) * (num_loops - 2), end])

    elif num_loops == 2:
        looped_signal = np.concatenate([start, end])

    else:
        raise BaseException("Error: Number of loops should be 2 or higher or it will not be a loop, duh!")

    output = cut(signal=looped_signal, goal_len=goal_len)  # Trim the looped signal
    return output


def loop_or_cut(signal, sr, duration_samples=None, duration_seconds=None, padding=0):
    '''
    Pads or truncates a signal to the desired duration.
    This function was inspired by: https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
    :param signal: input 1D signal
    :param duration_seconds: the duration that the signal needs to be resized to.
    :param sr: sample rate
    :param padding: what to use for padding if the signal is too short.
    :return:
    '''

    if duration_samples is None and duration_seconds is not None:
        goal_len = round(sr * duration_seconds)  # How many samples we want the file to be
    elif duration_seconds is None and duration_samples is not None:
        goal_len = round(duration_samples)
    else:
        raise ValueError("Either duration_seconds or duration_samples must be specified (not both, and at least one)")

    if len(signal) == goal_len:
        output = signal  # If the file is equal to the goal length nothing happens

    # If the signal is longer than goal_len it gets cut into a smaller signal with random cut points.
    # The cut points are random to make sure no patterns are stored in the cutting that the model will learn from.
    elif len(signal) > goal_len:
        output = cut(signal=signal, goal_len=goal_len)

    # If the file is shorter than goal_len it gets looped.
    else:
        output = loop(signal=signal, goal_len=goal_len)

    return output


def pad_or_cut(signal, duration_seconds=None, duration_samples=None, sr=SAMPLE_RATE, padding=0):
    '''
    Pads or truncates a signal to the desired duration.
    This function was inspired by: https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
    :param signal: input 1D signal
    :param duration_seconds: the duration that the signal needs to be resized to.
    :param sr: sample rate
    :param padding: what to use for padding if the signal is too short. np.nextafter(0,1) is used to get a
    number very close to zero, as it works better than 0 when using logs.
    :return:
    '''

    if duration_samples is None and duration_seconds is not None:
        goal_len = round(sr * duration_seconds)  # How many samples we want the file to be
    elif duration_seconds is None and duration_samples is not None:
        goal_len = round(duration_samples)
    else:
        raise ValueError("Either duration_seconds or duration_samples must be specified (not both, and at least one)")

    signal_len = len(signal)
    if signal_len == goal_len:
        output = signal  # If the file is equal to the goal length nothing happens

    # If the signal is longer than goal_len it gets cut into a smaller signal with random cut points.
    # The cut points are random to make sure no patterns are stored in the cutting that the model will learn from.
    elif signal_len > goal_len:
        cut_len = signal_len - goal_len
        cut_begin = random.randint(0, cut_len)  # Number of samples to cut of, in the beginning
        cut_end = cut_len - cut_begin  # Number of samples to cut off in the end
        output = signal[cut_begin:-cut_end]  # The first cut_begin samples and the last cut_end samples gets cut off.


    # If the file is shorter than goal_len it gets padded with zeroes both in the start end and end of the file.
    # The ratio of left/right padding is random, since we don't want any patterns in the padding that the model trains on.
    else:
        pad_len = goal_len - signal_len  # If goal_len is 30 sec and signal_len is 10 second we want 30-10 = 20 seconds padding
        pad_begin_len = random.randint(0,
                                       pad_len)  # Randomly select how long the padding should be in the beginning of the file
        pad_end_len = pad_len - pad_begin_len  # The remaining pad_len goes to the end of the file

        pad_begin = np.full(pad_begin_len, padding)  # Generates an array of zeroes for the beginning of the file
        pad_end = np.full(pad_end_len, padding)  # Generates an array of zeroes for the end of the file

        output = np.concatenate([pad_begin, signal, pad_end])

    return output


def looping(dataset_path, duration=30,
            sr=22050):  # Loops or shortens the audio files so they are all the same duration
    for (dir_path, dir_names, filenames) in os.walk(dataset_path):
        for filename in filenames:
            file = f"{dir_path}/{filename}"
            samples, sr = librosa.load(file, sr=SAMPLE_RATE)
            length_goal = sr * duration  # How many samples we want the files to be
            num_samples = len(samples)
            if num_samples >= length_goal:
                looped_file = samples.tolist()[:length_goal]  # If the file is longer than the goal, we shorten it
            else:
                factor = math.ceil(
                    length_goal / num_samples)  # Produces the smallest factor we can multiply with to get above the length_goal.
                looped_file = (samples.tolist() * factor)[
                              :length_goal]  # We loop the file with the factor, and the cut of the excess.
            return looped_file


def find_longest_mp3(dataset_path):
    max_duration = 0
    max_file = None
    for (dirpath, dirnames, filenames) in os.walk(dataset_path):
        print(f"Running through:  {dirpath}")
        for filename in filenames:
            file = f"{dirpath}/{filename}"
            duration = MP3(file).info.length
            if duration > max_duration:
                max_duration = duration
                max_file = file
    print(f"Max file:  {max_file}")
    print(f"Max duration:  {max_duration}")
    return max_file, max_duration
