import sys
import time
import numpy as np
import preprocessing
from pathlib import Path
import files
from mne.io import read_raw_ctf
from mne import find_events, annotations_from_events, set_log_level


if __name__ == '__main__':

    try:
        index = int(sys.argv[1])
    except:
        raise IndexError("no file index")

    try:
        json_file = sys.argv[2]
    except:
        json_file = "settings.json"

    set_log_level("WARNING")
    
    start_time = time.time()

    settings = preprocessing.load_json(json_file)

    raw_path = Path(settings["dataset_path"]).joinpath("raw")
    processed_path = Path(settings["dataset_path"]).joinpath("processed")

    all_dirs = files.get_directories(raw_path, strings=".ds", check="all")
    all_dirs = [i for i in all_dirs if not i.name == "hz.ds"]
    all_dirs.sort()

    raw_dir = all_dirs[index]

    subject = raw_dir.parts[-2]

    dataset = raw_dir.parts[-1].split("_")[-1].split(".")[0].zfill(3)

    status = "START"
    to_print = f"{status} File:{str(index+1).zfill(3)}/{len(all_dirs)} Subject:{subject} Dataset:{dataset}"
    if subject in settings["excluded_subs"]:
        status = "EXCLUDED"
        to_print = f"{status} File:{str(index+1).zfill(3)}/{len(all_dirs)} Subject:{subject} Dataset:{dataset}"
        print(to_print)
        quit()
    
    print(to_print)
    
    # prepare directory

    subject_path = files.make_directory(processed_path, subject)

    subject_raw_file = f"sub-{subject}-treasure_hunt-block_{dataset}-raw.fif"
    subject_file_info = f"sub-{subject}-treasure_hunt-block_{dataset}-file_info.json"

    raw = read_raw_ctf(
        raw_dir, clean_names=True, verbose=False, preload=True
    )

    # not every recording has EOGs
    try:
        set_ch = {
            "EEG057":"eog", 
            "EEG058": "eog",
            "EEG059": "eog",
            "EEG060": "eog",
            "UPPT001": "stim", 
            "UPPT002": "stim"}
        
        raw = raw.set_channel_types(set_ch)

    except:
        set_ch = {
            "UPPT001": "stim", 
            "UPPT002": "stim"}
        
        raw = raw.set_channel_types(set_ch)

    events = find_events(raw, stim_channel="UPPT001", shortest_event=1)
    key_presses = find_events(raw, stim_channel="UPPT002", shortest_event=1)

    trigger_mapping = preprocessing.load_json(settings["trigger_mapping"])
    key_press_mapping = preprocessing.load_json(settings["key_press_mapping"])

    # the down key press had a value 1 that would conflict with a weird trigger 1
    key_press_mapping["7"] = "down"
    key_press_mapping.pop("1")
    key_presses[:,2][key_presses[:,2] == 1] = 7

    # stack key presses and triggers
    total_triggers = np.vstack([events, key_presses])
    total_triggers = total_triggers[total_triggers[:, 0].argsort()]

    # stack mapping dicts
    total_mapping = {**trigger_mapping, **key_press_mapping}

    total_mapping = {int(key): value for key, value in total_mapping.items()}

    total_annotations = annotations_from_events(
        total_triggers, sfreq=raw.info["sfreq"], event_desc=total_mapping
    )

    raw = raw.set_annotations(total_annotations)

    # trimming to +/- 2s around block start/break triggers
    tmin = 0.0
    tmax = None
    try:
        begin_annot = [i for i in raw.annotations.count().keys() if any([i == j for j in ["block_start", "experiment_start"]])]
        end_annot = [i for i in raw.annotations.count().keys() if any([i == j for j in ["block_break", "experiment_end"]])]

        if len(begin_annot) > 0:
            begin = np.min([i["onset"] for i in raw.annotations if i["description"] == begin_annot[0]])
            tmin = begin - 2.0
            if tmin < 0:
                tmin = 0.0
        
        if len(end_annot) > 0:
            end = np.max([i["onset"] for i in raw.annotations if i["description"] == end_annot[0]][0])
            tmax = end + 2.0
            if tmax > raw.times[-1]:
                tmax = None
    
        raw = raw.crop(
            tmin=tmin, tmax=tmax
        )
    
    except:
        tmin = 0.0
        tmax = None
        
        raw = raw.crop(
            tmin=tmin, tmax=tmax
        )

    raw = raw.filter(None, 120.0, picks=["meg"])

    raw.save(subject_path.joinpath(subject_raw_file), fmt="single", overwrite=True)

    status = "END"
    time_elapsed =  np.round((time.time() - start_time)/60, 2)
    to_print = f"{status} File:{str(index+1).zfill(3)}/{len(all_dirs)} Subject:{subject} Dataset:{dataset} Time elapsed: {time_elapsed} min"
    print(to_print)


