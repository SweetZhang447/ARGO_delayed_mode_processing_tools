from pathlib import Path

import numpy as np
from delayed_mode_processing import first_time_run
from tools import read_intermediate_nc_file,  make_intermediate_nc_file

def check_NETCDF_format():
    check_dir = Path(r"c:\Users\szswe\Desktop\FLOAT_DATA\F9444\DMODE\CHECK")

    rejected = []
    for f in sorted(check_dir.glob("*.filecheck")):
        if "<status>FILE-REJECTED</status>" in f.read_text():
            rejected.append(f.stem)  # e.g. D2904018_001.nc

    print(f"Rejected files ({len(rejected)}):")
    for name in rejected:
        print(f"  {name}")

def run_counts():
    data_dir = Path(r"c:\Users\szswe\Desktop\FLOAT_DATA\F10051\DMODE\F10051_VI")
    float_num = "1902655"
    argo_data = read_intermediate_nc_file(data_dir)

    count_mask = np.logical_or(argo_data["NB_SAMPLE_CTD"] > 100, np.logical_and(argo_data["NB_SAMPLE_CTD"] < 1, argo_data["NB_SAMPLE_CTD"] != -99))

    # For count_mask levels, confirm the spike with a large PSAL gradient to either neighbor.
    psal = argo_data["PSAL_ADJUSTED"]
    diffs = np.abs(np.diff(psal, axis=1))                                   # (n_profiles, n_levels-1)
    diff_forward  = np.pad(diffs, ((0, 0), (0, 1)), constant_values=np.nan) # diff to next level
    diff_backward = np.pad(diffs, ((0, 0), (1, 0)), constant_values=np.nan) # diff to prev level
    gradient_mask = (diff_forward > 15) | (diff_backward > 15)

    final_mask = count_mask & gradient_mask
    argo_data["PSAL_ADJUSTED_QC"][final_mask] = 4
    # For each profile: if the last valid point is in count_mask AND the last two
    # PSAL_ADJUSTED points differ by >= 0.02 PSU, mark the last point as bad.
    n_profiles, n_levels = psal.shape
    for i in range(n_profiles):
        valid_idx = np.where(~np.isnan(psal[i]))[0]
        if len(valid_idx) < 2:
            continue
        last, second_last = valid_idx[-1], valid_idx[-2]
        if count_mask[i, last] and abs(psal[i, last] - psal[i, second_last]) >= 0.02:
            argo_data["PSAL_ADJUSTED_QC"][i, last] = 4

    make_intermediate_nc_file(argo_data, data_dir, float_num)


run_counts()
