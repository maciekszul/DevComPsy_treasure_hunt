import sys
import time
import files
import preprocessing
from pathlib import Path
from mne import set_log_level
from mne.io import read_raw
from mne.preprocessing import read_ica


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

    processed_path = Path(settings["dataset_path"]).joinpath("processed")

    fif_paths = files.get_files(processed_path, "*.fif", strings=["zapline", "treasure_hunt", "-raw.fif"])

    key = fif_paths[index]
    misc_fif_files = files.get_files(key.parent, "*.fif", strings=key.stem.split("-")[:-1])
    misc_json_files = files.get_files(key.parent, "*.json", strings=key.stem.split("-")[:-1] + [])
    misc_files = misc_fif_files + misc_json_files
    misc_files.sort()

    ecg_score, ica_path, raw_path = misc_files

    raw = read_raw(raw_path, preload=True)
    raw = raw.crop(tmin=0, tmax=100).filter(5, 35)

    ica = read_ica(ica_path)
    ica.plot_sources(raw, block=True)

    print(key.name, "excluded comps: ", ica.exclude)

    ica.save(ica_path, overwrite=True)
    print("saved:", ica_path)

    print(misc_files)
