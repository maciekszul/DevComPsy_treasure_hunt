import sys
import time
import files
import preprocessing
import numpy as np
from pathlib import Path
from mne import set_log_level
from mne.io import read_raw_fif
from mne.preprocessing import read_ica
from scipy.stats import spearmanr


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

    settings = preprocessing.load_json(json_file)

    processed_path = Path(settings["dataset_path"]).joinpath("processed")

    fif_paths = files.get_files(processed_path, "*.fif", strings=["zapline", "treasure_hunt", "-raw.fif"])
    ica_paths = files.get_files(processed_path, "*.fif", strings=["zapline", "treasure_hunt", "-ica.fif"])
    json_paths = files.get_files(processed_path, "*.json", strings=["treasure_hunt", "file_info"])

    fif_path = fif_paths[index]
    subject_no = fif_path.stem.split("-")[1]
    block = fif_path.stem.split("-")[3]
    
    ica_path = [i for i in ica_paths if all([j in i.stem for j in [subject_no, block]])][0]
    json_path = [i for i in json_paths if all([j in i.stem for j in [subject_no, block]])][0]

    file_info = preprocessing.load_json(json_path)

    if not file_info.get("ica_rejected") == None:
        print("ICA CHECKED")
        sys.exit()

    print(ica_path.name)

    raw = read_raw_fif(fif_path, preload=True)
    raw = raw.crop(tmin=100, tmax=200).filter(1, 20)
    ica = read_ica(ica_path)

    eog_comps = []

    try:
        eogs = raw.copy().pick(["eog"]).get_data()
        ica_comps = ica.get_sources(raw).get_data()
        rho_eog_ica = []
        pval_eog_ica = []
        for i_c in ica_comps:
            rho_eog_ica.append([np.array(spearmanr(i_c, i)).astype(float)[0] for i in eogs])
            pval_eog_ica.append([np.array(spearmanr(i_c, i)).astype(float)[1] for i in eogs])

        rho_eog_ica = np.array(rho_eog_ica)
        pval_eog_ica = np.array(pval_eog_ica)

        eog_comps = np.arange(ica.n_components)[np.apply_along_axis(any, 1, rho_eog_ica > 0.4)].tolist()
    except:
        pass
    
    if len(eog_comps) > 0:
        ica.exclude.append(eog_comps)
    else:
        pass

    ica.plot_components()
    ica.plot_sources(raw, block=True, theme="dark")

    print(fif_path.name, "excluded comps: ", ica.exclude)

    preprocessing.update_json_file(
        json_path, {"ica_rejected": ica.exclude}
    )

    ica.save(ica_path, overwrite=True)
    print("saved:", ica_path.name)