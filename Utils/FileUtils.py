import glob
import hashlib
import png
import os
import io
import random
import shutil
from urllib.request import urlopen
import pathlib
import numpy as np
import pandas as pd
import requests
import re
import skimage.io
from keras.models import load_model


# Creates extra columns in the dataframe which are "helper columns" used storing the files in the correct folders and as the correct filename.
# save_whale_id determines if a "whale_id" should be a part of the file name. "whale_id" is the id of a specific whale individual. "id" is the id of a specific recording.
# Suffix is the file extension the files should have.
def prepdata(dataset, suffix, save_whale_id=False):
    if save_whale_id:
        whalefm = dataset.dropna(subset=["location", "whale_type", "whale_id"])
        tmp = whalefm.loc[:, "whale_type"].str.split(pat=" ", expand=True).copy()
        whalefm.loc[:, "dir"] = tmp.loc[:, 0]
        whalefm['file_name'] \
            = whalefm.loc[:, 'id'].apply(str) + "_" + whalefm.loc[:, 'whale_type'] \
            .str.replace(' ', '') + "_" + whalefm.loc[:, "whale_id"].apply(str) + suffix
    else:
        whalefm = dataset.dropna(subset=["location", "whale_type"])
        tmp = whalefm.loc[:, "whale_type"].str.split(pat=" ", expand=True).copy()
        whalefm.loc[:, "dir"] = tmp.loc[:, 0]
        whalefm.loc[:, "file_name"] = whalefm.loc[:, "id"].apply(str) + "_" + whalefm.loc[:, 'whale_type'].str.replace(
            ' ', '') + suffix
    return whalefm


# Converts mp3 files in from_dir to wav files in to_dir
def convert_mp3dir_to_wav(from_dir, to_dir, overwrite=False):
    os.system("ffmpeg -loglevel 0")
    walker = os.walk(from_dir)
    for (dir, dirnames, filenames) in walker:
        if len(filenames) != 0:
            print(f"Converting files in {dir}")
            for filename in filenames:
                filepath = os.path.join(dir, filename)
                subfolder = re.split('_|\.', filename)[1]
                filename_wav = filename.split(".")[0] + ".wav"
                destination = os.path.join(to_dir, subfolder, filename_wav)

                convert_mp3file_to_wav(filepath, destination)


#  Converts a single mp3 to wav
def convert_mp3file_to_wav(filepath, destination):
    cmd = f"ffmpeg -i {filepath} -vn -ar 22050 {destination}"
    os.system(cmd)


# save_whale_id saves the id belonging to a specific individual whale in the file name. Otherwise it just saves
# the id of the recording
def save_data_from_dataframe(dir_path, dataframe, save_whale_id=False, suffix='.mp3'):
    whalefm = prepdata(dataframe, save_whale_id, suffix)
    whalefm.reset_index()
    for index, row in whalefm.iterrows():
        url = row.location
        path = pathlib.Path(f'{dir_path}/{row.dir}/{row.file_name}')
        download_and_save_file(url, path)


# Simple way to download and save a file from a URL. Tested and works with mp3
def download_and_save_file(path, url):
    with open(path, "wb") as f:
        f.write(requests.get(url).content)

#  Scales array to have values from 0-max
def img_scaling(array, max):
    array = array + abs(array.min())  # To eliminate negative values
    array = array * (max / array.max())  # Scale so that highest number in array = max

    return array.astype(np.uint8)


# Saves a
def save_img_from_array(array, bit_depth, destination):
    #  https://stackoverflow.com/questions/25696615/can-i-save-a-numpy-array-as-a-16-bit-image-using-normal-enthought-python/25814423#25814423

    if bit_depth == 16:
        arr = img_scaling(array, 65535).astype(np.uint16)  # scales to 16 bit (0 to 65535)

        with open(destination, 'wb') as f:
            writer = png.Writer(width=arr.shape[1], height=arr.shape[0], bitdepth=16, greyscale=True)
            writer.write(f, arr.tolist())

    elif bit_depth == 8:
        arr = img_scaling(array, 255)  # scales to 8 bit (0 to 255)
        a=1
        # save as PNG
        skimage.io.imsave(destination, arr)

    else:
        raise BaseException(f"Could not infer bitdepth = {bit_depth}. Function only works with values 8 or 16")

def load_image(path):
    spec = skimage.io.imread(path, as_gray=True)
    return spec

def save_spec_as_array(spec, destination):
    np.save(destination, spec)

def load_spec(array_path):
    spec = np.load(array_path)
    return spec


# From https://www.youtube.com/watch?v=XrALbgIbHzc&ab_channel=CodeBear
# Deletes duplicate files in all folders and subfolders of the dir_path
def delete_duplicate_files(dir_path):
    walker = os.walk(dir_path)
    unique_files = dict()
    for (dir, dirnames, filenames) in walker:
        print(len(filenames))
        if len(filenames) != 0:
            print(f"Searcing {dir}")
            for file in filenames:
                filepath = os.path.join(dir, file)
                filehash = hashlib.md5(open(filepath, "rb").read()).hexdigest()

                if filehash in unique_files.keys():
                    os.remove(filepath)
                    print(f"Deleted {filepath}")
                else:
                    unique_files[filehash] = filepath

    print("delete_duplicate_files done")


# Makes a list of ids of all the files in dir_path and filters everything else out of the dataframe
# Useful if for example some duplicates have been deleted from the data, but not in the dataframe
# Requires that the files have id's that are in the dataframe
def match_csv_to_dir(dir_path, csv_path):
    df = pd.read_csv(csv_path)
    len_before = len(df.id)
    ids = []  # List of id's in the dir
    walker = os.walk(dir_path)
    for (dir, dirnames, filenames) in walker:
        if dir != dir_path:  # Makes sure not to include files in the root dir
            print(f"Processing {len(filenames)} files in {dir}")
            for file in filenames:
                id = file.split("_")[0]
                ids.append(int(id))

    df = df.loc[df["id"].isin(ids), :]

    # These should match:
    if len(ids) != len(df):
        raise BaseException("Error: Length of dataframe does not match number of files processed")

    df = df.reset_index(drop=True)

    print("match_csv_to_dir done")

    return df




# Splits a root dir into train, val and test dirs. Works with all formats.
# Requires: Filenames in the folders needs to contain label name and match the glob search string.
def dir_split(root_dir, labels: list, train_fraction=0.55, val_fraction=0.25, test_fraction=0.20, glob_string="*{}*",
              equal_dist=False):
    """
    :param root_dir: the directory where all the files reside
    :param train_fraction: fraction of the files that are used for training from 0-1
    :param val_fraction: fraction of the files that are used for validation from 0-1
    :param test_fraction: fraction of the files that are used for testing from 0-1
    :param labels: labels of the classes in the data. The function expects that labels are in the filenames.
    :param glob_string: a search string used to search for label names in files. use {} to specify where in the filename
    the label name should be searched for. for example: "{}*" will search for files with label name in the beginning of
    the filename.
    :param equal_dist: determines if distribution between labels is equal. Is not fancy: The label with the fewest data
    determines size of sets, and the bigger sets gets smaller.
    :return: Nothing to return
    """

    # Makes a copy of the dir
    if os.path.isdir(root_dir + "_fresh") is False:
        print("Making a copy of root_dir...")
        shutil.copytree(root_dir, root_dir + "_fresh")

    assert train_fraction + val_fraction + test_fraction == 1, "train_part, val_part and test_part should have sum = 1"

    os.chdir(root_dir)
    if os.path.isdir(f"train/{labels[0]}") is False:  # If the train folder already exists we don't want to do the split
        #  Make the split dirs
        for label in labels:
            os.makedirs(f"train/{label}")
        for label in labels:
            os.makedirs(f"val/{label}")
        for label in labels:
            os.makedirs(f"test/{label}")

        if equal_dist:  # TODO: This needs to be tested:
            min_len = float("inf")

            #  Find the smallest dataset and make the split sizes according to it
            for label in labels:
                g = glob.glob(glob_string.format(label))
                if len(g) > min_len:
                    min_len = len(g)
            train_size = min_len * train_fraction
            val_size = min_len * val_fraction
            test_size = min_len * test_fraction
            assert train_size + val_size + test_size == min_len

            for label in labels:
                #  Put all files that have label in the middle somewhere into a list
                label_data = glob.glob(glob_string.format(label))

                # move the train folder
                for c in random.sample(label_data, int(train_size)):
                    shutil.move(c, f"train/{label}")

            for label in labels:
                label_data = glob.glob(glob_string.format(label))

                # move the val folder
                for c in random.sample(label_data, int(val_size)):
                    shutil.move(c, f"val/{label}")

            for label in labels:
                label_data = glob.glob(glob_string.format(label))

                # move to the test folder:
                for c in random.sample(label_data, int(test_size)):
                    shutil.move(c, f"test/{label}")

        else:
            sizes = {label: {} for label in labels}

            for label in labels:
                label_data = glob.glob(glob_string.format(label))

                # Compute the relevant train, val and test sizes for each label
                sizes[label]["train"] = int(train_fraction * len(label_data))
                sizes[label]["test"] = int(test_fraction * len(label_data))
                sizes[label]["val"] = int(val_fraction * len(label_data))

                # move the train folder
                for c in random.sample(label_data, sizes[label]["train"]):
                    shutil.move(c, f"train/{label}")

            for label in labels:
                label_data = glob.glob(glob_string.format(label))

                # move the val folder
                for c in random.sample(label_data, sizes[label]["val"]):
                    shutil.move(c, f"val/{label}")

            for label in labels:
                label_data = glob.glob(glob_string.format(label))

                # move to the test folder:
                for c in random.sample(label_data, sizes[label]["test"]):
                    shutil.move(c, f"test/{label}")

    else:
        raise BaseException("Train folder already exists")

    print("Dirsplit completed succesfuly")


def save_model_to_h5(model, path):
    model.save(path)

def load_h5_model(path):
    return load_model(path)

# Some alternative way to download. Dont use this as standard
def download_and_save_file_io(path, url):
    z = io.BytesIO(urlopen(url).read())
    data = z.getbuffer().tobytes()
    with open(path, mode='xb') as f:
        f.write(data)



