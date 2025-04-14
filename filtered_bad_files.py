import librosa
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm
from glob import glob

import os
import argparse


def load_audio(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        if len(audio) == 0:
            raise ValueError("Audio file is empty")
        return file_path
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def scan_cache(save_dir):
    cache_files = glob(os.path.join(save_dir, "*.wav"))
    return cache_files

def remove_file(file_path):
    try:
        os.remove(file_path)
        print(f"Removed file: {file_path}")
    except Exception as e:
        print(f"Error removing file {file_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter bad audio files.")
    parser.add_argument("--save_dir", type=str, default="./data", help="Directory to save the filtered audio files.")
    args = parser.parse_args()

    save_dir = args.save_dir

    cache_files = scan_cache(save_dir)

    with ThreadPool(16) as pool:
        results = list(tqdm(pool.imap(load_audio, cache_files), total=len(cache_files)))
    bad_files = []

    for file_path in results:
        if file_path is None:
            bad_files.append(file_path)

    for file in bad_files:
        remove_file(file)
    print(f"Removed {len(bad_files)} bad files.")




    

