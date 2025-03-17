import sys
import json
import time
import preprocessing
import files
import numpy as np
import matplotlib.pylab as plt
from copy import copy
from pathlib import Path
from ecgdetectors import Detectors
from mne import set_log_level
from mne.io import read_raw_fif, RawArray
from mne.preprocessing import EOGRegression, ICA
from mne.time_frequency import psd_array_welch
from meegkit.dss import dss_line_iter


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

    raw_fif_paths = files.get_files(processed_path, "*.fif", strings=["treasure_hunt", "-raw.fif"])
    raw_fif_paths = [i for i in raw_fif_paths if "zapline" not in i.stem]
    raw_fif_paths.sort()

    raw_fif_path = raw_fif_paths[index]

    file_info_path = files.get_files(
        raw_fif_path.parent, "*.json", 
        strings=raw_fif_path.stem.split("-")[:-1]
    )[0]

    subject = raw_fif_path.parts[-2]

    dataset = raw_fif_path.parts[-1].split("_")[-1].split(".")[0].zfill(3)

    status = "START"
    to_print = f"{status} File:{str(index+1).zfill(3)}/{len(raw_fif_paths)} Subject:{subject} Dataset:{dataset}"
    if subject in settings["excluded_subs"]:
        status = "EXCLUDED"
        to_print = f"{status} File:{str(index+1).zfill(3)}/{len(raw_fif_paths)} Subject:{subject} Dataset:{dataset}"
        print(to_print)
        quit()

    print(to_print)

    subject_path = raw_fif_path.parent

    raw = read_raw_fif(raw_fif_path, verbose=False, preload=True)
    raw_data = raw.get_data()
    raw_info = raw.info
    first_samp = raw.first_samp
    annotations = raw.annotations
    mag_ix = np.array([i for i, lab in enumerate(raw.get_channel_types()) if lab == "mag"])

    raw_psds, freqs = psd_array_welch(
        raw.copy().pick("mag").get_data(), n_fft=1024,
        sfreq=600.0, fmin=0, fmax=100, average="mean"
    )

    rd = np.array_split(raw_data[mag_ix], 10, axis=1)
    rd = [np.moveaxis(i, [0,1], [1,0]) for i in rd]
    rd_f = []

    for ix, ss in enumerate(rd):
        print(raw_fif_path.name, f"{ix+1}/10")
        data, iters = dss_line_iter(ss, fline=50.0, sfreq=raw_info["sfreq"], spot_sz=5.5, win_sz=10, nfft=1024)
        rd_f.append(data)
    del rd
    rd_f = [np.moveaxis(i, [0,1], [1,0]) for i in rd_f]
    rd_f = np.hstack(rd_f)

    new_raw_data = copy(raw_data)
    del raw_data
    new_raw_data[mag_ix, :] = rd_f

    new_raw = RawArray(
        new_raw_data,
        raw_info,
        first_samp=first_samp
    )
    new_raw = new_raw.set_annotations(annotations)

    zap_psds, freqs = psd_array_welch(
        new_raw.copy().pick("mag").get_data(), n_fft=1024,
        sfreq=600.0, fmin=0, fmax=100, average="mean"
    )

    add = {
        "raw": raw_psds.astype(np.single).tolist(),
        "raw_zapline": zap_psds.astype(np.single).tolist(),
        "freqs": freqs.astype(np.single).tolist()
    }

    # EOG
    
    try:
        model_eog = EOGRegression(picks="mag", picks_artifact="eog").fit(new_raw)
        new_raw = model_eog.apply(new_raw)
        
        qt_path = files.make_directory(subject_path, "quality")
        eog_file = "-".join(raw_fif_path.stem.split("-") + ["EOG_regression.jpeg"])
        plt.ioff()
        f, ax = plt.subplots(4, 1, figsize=(4, 8))
        model_eog.plot(axes=ax, show=False)
        f.savefig(qt_path.joinpath(eog_file), dpi=90)
        plt.close("all")

        new_raw_path = raw_fif_path.parent.joinpath("zapline_" + raw_fif_path.name)
        new_raw.save(new_raw_path, fmt="single", overwrite=True)
    except:
        pass


    # ICA
    new_raw = new_raw.filter(1,30, verbose=False)
    n_ica = 20
    ica = ICA(n_components=n_ica)
    ica.fit(new_raw)
    ica_data = ica.get_sources(new_raw).get_data()

    sfreq = new_raw.info["sfreq"]

    results_dict = {}
    for ica_comp in range(n_ica):
        detector = Detectors(sfreq)
        r_peaks = detector.swt_detector(ica_data[ica_comp])
        pos_r_peaks = preprocessing.adjust_QRS_peaks(ica_data[ica_comp], r_peaks, 100, positive=True)
        neg_r_peaks = preprocessing.adjust_QRS_peaks(ica_data[ica_comp], r_peaks, 100, positive=False)
        pos_med_diff = np.median(np.diff(pos_r_peaks) / sfreq)
        neg_med_diff = np.median(np.diff(neg_r_peaks) / sfreq)
        results_dict[ica_comp] = [pos_r_peaks.tolist(), neg_r_peaks.tolist(), pos_med_diff, neg_med_diff]

    ica_filename = new_raw_path.parent.joinpath(new_raw_path.stem[:-4] + "-ica.fif")
    ica.save(ica_filename, overwrite=True, verbose=False)

    json_filename = new_raw_path.parent.joinpath(new_raw_path.stem[:-4] + "-ecg_score.json")

    preprocessing.save_dict_as_json(subject_path.joinpath(json_filename), results_dict)

    status = "END"
    time_elapsed =  np.round((time.time() - start_time)/60, 2)
    to_print = f"{status} File:{str(index).zfill(3)}/{len(raw_fif_paths)} Subject:{subject} Dataset:{dataset} Time elapsed: {time_elapsed} min"
    print(to_print)

