import sys
import time
import numpy as np
import preprocessing
from pathlib import Path
import files
from mne.io import read_raw_ctf
from mne import set_log_level


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
    beh_model_dir = Path(settings["dataset_path"]).joinpath("old_analysis", "beh_model")

    all_dirs = files.get_directories(raw_path.joinpath("MEG"), strings=".ds", check="all")
    all_dirs = [i for i in all_dirs if not i.name == "hz.ds"]
    all_dirs.sort()

    all_beh = files.get_files(raw_path, "*.mat", strings=["_beh"])
    all_beh_model = files.get_files(beh_model_dir, "*.mat", strings=["_beh_model"])

    raw_dir = all_dirs[index]

    subject = raw_dir.parts[-2]

    dataset = raw_dir.parts[-1].split("_")[-1].split(".")[0].zfill(3)

    status = "START"
    to_print = f"{status} File:{str(index).zfill(3)}/{len(all_dirs)} Subject:{subject} Dataset:{dataset}"
    if subject in settings["excluded_subs"]:
        status = "EXCLUDED"
        to_print = f"{status} File:{str(index).zfill(3)}/{len(all_dirs)} Subject:{subject} Dataset:{dataset}"
        print(to_print)
        quit()
    
    print(to_print)

    subject_path = files.make_directory(processed_path, subject)

    beh_file = [i for i in all_beh if subject in i.stem][0]
    beh_model_file = [i for i in all_beh_model if subject in i.stem][0]

    beh_df = preprocessing.mat_beh_as_df(beh_file)
    beh_model_df = preprocessing.mat_beh_as_df(beh_model_file)

    subject_file_info = f"sub-{subject}-treasure_hunt-block_{dataset}-file_info.json"

    subject_beh_file = f"sub-{subject}-treasure_hunt-beh.csv"
    subject_beh_model_file = f"sub-{subject}-treasure_hunt-beh_model.csv"

    subject_beh_path = subject_path.joinpath(subject_beh_file)
    subject_model_file_path = subject_path.joinpath(subject_beh_model_file)

    raw = read_raw_ctf(
        raw_dir, clean_names=True, verbose=False, preload=False
    )

    file_info = {}

    mcn, mcix = preprocessing.find_missing_channels(raw)

    file_info["missing_channel"] = list(mcn)
    file_info["missing_channel_ix"] = [int(i) for i in mcix]
    try:
        file_info["stim_channels"] = raw.copy().pick_types(stim=True).ch_names
    except:
        file_info["stim_channels"] = [i for i in raw.ch_names if any([i == j for j in ["UPPT001", "UPPT002"]])]

    try:
        file_info["eog"] = [i for i in raw.ch_names if "EEG" in i]
    except:
        file_info["eog"] = []

    file_info["beh_file"] = str(subject_beh_path)
    file_info["beh_model"] = str(subject_model_file_path)

    beh_df.to_csv(subject_beh_path, index=False)
    beh_model_df.to_csv(subject_model_file_path, index=False)

    preprocessing.save_dict_as_json(subject_path.joinpath(subject_file_info), file_info)

    status = "END"
    time_elapsed =  np.round((time.time() - start_time)/60, 2)
    to_print = f"{status} File:{str(index).zfill(3)}/{len(all_dirs)} Subject:{subject} Dataset:{dataset} Time elapsed: {time_elapsed} min"
    print(to_print)

