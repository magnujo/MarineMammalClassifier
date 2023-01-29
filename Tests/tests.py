import os
import io
from urllib.request import urlopen
from pathlib import Path


# test if a root dir contains as many files as there are rows in a dataframe
def files_count_test(dir, dataframe):
    count = 0
    for root_dir, cur_dir, files in os.walk(dir):
        count = count + len(files)
    return count == dataframe.shape[0]

# test if a dir1 contains as many files as dir2
def files_count_test_dirs(dir1, dir2):
    count1 = 0
    count2 = 0
    for root_dir, cur_dir, files in os.walk(dir1):
        count1 = count1 + len(files)
    for root_dir, cur_dir, files in os.walk(dir2):
        count2 = count2 + len(files)
    return count1 == count2


# Test if the byte of the saved files is the same as the online files
def byte_size_test(dir, dataframe):
    for root_dir, cur_dir, files in os.walk(dir):
        print(f"Root: {root_dir}")
        print(f"cur: {cur_dir}")
        print(f"files: {len(files)}")
        for count, file in enumerate(files):
            print(count)
            i = int(file.split("_")[0])

            url = dataframe.loc[dataframe.id == i, "location"].iloc[0]

            z = io.BytesIO(urlopen(url).read())
            burl = z.getbuffer().tobytes()
            bfile = Path(f"{root_dir}/{file}").read_bytes()

            if len(burl) != len(bfile):
                return False
        print(f"Done with {cur_dir}")
    return True




