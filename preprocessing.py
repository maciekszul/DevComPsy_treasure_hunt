import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
from mne.io import read_raw_ctf
from mne.channels import read_layout
from mne import find_events, annotations_from_events


def load_json(json_path):
    with open(json_path) as json_file:
        data = json.load(json_file)
    return data


def find_missing_channels(raw, layout="CTF275", ch_name_string="M"):
    """Returns missing channels and the indices"""
    lay_full = read_layout(fname=layout)
    all_chan = lay_full.names
    raw_chan = [i for i in raw.info["ch_names"] if i[0] == ch_name_string]
    missing_chan = list(set(all_chan) - set(raw_chan))
    missing_chan_ix = lay_full.pick(picks=missing_chan).ids

    return missing_chan, missing_chan_ix


def mat_beh_as_df(path):
    """Extracts behavioral data, and returns it as a pandas DataFrame"""
    data = loadmat(path, simplify_cells=True)
    headers = data["beh"]["descr"]
    content = data["beh"]["dat"]
    data = pd.DataFrame.from_dict({h: content[:,ix] for ix, h in enumerate(headers)})
    return data


def save_dict_as_json(file_path, dictionary):
    """Saves a dictionary as a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(dictionary, json_file, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving dictionary: {e}")


def update_json_file(file_path, update_dict):
    """Updates an existing JSON file with a dictionary. Replaces values of existing keys."""
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        
        data.update(update_dict)
        
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error updating dictionary: {e}")


def adjust_QRS_peaks(signal, peaks, half_window, positive=True):
    new_peaks = np.zeros(len(peaks))
    for peak_ix, peak in enumerate(peaks):
        start_ix = peak - half_window
        if start_ix < 0:
            start_ix = 0
                    
        end_ix = peak + half_window
        signal_slice = signal[start_ix:end_ix]
        
        if positive == True:
            slice_max_ix = np.argmax(signal_slice)
        else:
            slice_max_ix = np.argmin(signal_slice)
        old_new_dist = slice_max_ix - half_window
        new_peak = peak + old_new_dist
        new_peaks[peak_ix] = new_peak
    new_peaks = np.unique(new_peaks).astype(int)
    return new_peaks