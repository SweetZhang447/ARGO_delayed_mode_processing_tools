"""
drift_analysis.py — Salinity drift analysis for ARGO floats.

Compares Argo float salinity against independent reference datasets
(ship CTDs, AXCTDs, bottle samples, GEM data, and nearby float profiles)
to detect and quantify salinity drift over time.

Two comparison methods are used:
  1. Average PSAL at a fixed pressure range (e.g. 500–600 dbar)
  2. PSAL interpolated to a target isotherm temperature (e.g. 2°C)

Reference data readers (also available in read_ctds.py for ship CTD data):
  read_ORP_WOOD_ctd, read_nicole_ctd, read_AXCTDs, read_regular_file,
  read_corrected_gem_data, read_bottle_data, read_lorenze_ctd_data, read_2903449 (via read_ctds.py)

Float-specific analysis functions:
  generate_F9185_F9444_avg_PSAL / PSAL_at_TEMP
  generate_F10052_avg_PSAL / PSAL_at_TEMP
  F9444_avg_PSAL / PSAL_AT_TEMP / TS
  F9443_avg_PSAL / PSAL_AT_TEMP / TS

Helper utilities:
  dm_to_decimal, filter_pres_levels, read_float_apply_qc,
  filter_float_overlap_date_range, find_psal_at_temp, make_TS_plot

All Julian days referenced to 1950-01-01 00:00:00 UTC.
Hardcoded file paths reference C:\\Users\\szswe\\Desktop\\...
"""
from datetime import datetime, timedelta
import itertools
import os
from pathlib import Path
import gsw
from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
from delayed_mode_processing import interpolate_missing_julian_days
from tools import from_julian_day, read_intermediate_nc_file, to_julian_day

######## HELPER FUNCTIONS ########
# Convert degrees + minutes to decimal degrees
def dm_to_decimal(deg, minutes, hemisphere):
    """
    Convert degrees and decimal minutes to decimal degrees.

    Parameters
    ----------
    deg : float
        Degrees component.
    minutes : float
        Decimal minutes component.
    hemisphere : str
        'N', 'S', 'E', or 'W'. South and West are returned as negative values.

    Returns
    -------
    float
        Decimal degrees (negative for S/W).
    """
    decimal = deg + minutes / 60.0
    if hemisphere in ("S", "W"):
        decimal *= -1
    return decimal
def filter_pres_levels(float1_data, pres_min, pres_max):
    """
    Set all values outside a pressure range to NaN across all data arrays.

    Modifies PSALs, TEMPs, PRESs, and CNDCs (if present) in place on the
    returned dict. Values where pressure < pres_min or > pres_max become NaN.

    Parameters
    ----------
    float1_data : dict
        Float data dict with keys PRESs, PSALs, TEMPs (and optionally CNDCs).
    pres_min, pres_max : float
        Pressure bounds (inclusive) in dbar.

    Returns
    -------
    float1_data : dict
        Same dict with out-of-range values set to NaN.
    """
    float1_pres_filter = (float1_data["PRESs"] >= pres_min) & (float1_data["PRESs"] <= pres_max)
    for var in ["PRESs", "PSALs", "TEMPs"]:
        float1_data[var][~float1_pres_filter] = np.nan

    return float1_data
def read_float_apply_qc(nc_filepath):
    """
    Read intermediate netCDF files and apply QC masking.

    Reads via read_intermediate_nc_file(), then sets all PSAL, PRES, and TEMP
    values to NaN where QC = 3 or 4 (probably bad or bad). Also renames
    PSAL_ADJUSTED → PSALs, PRES_ADJUSTED → PRESs, TEMP_ADJUSTED → TEMPs for
    consistency with the reference data dicts.

    Parameters
    ----------
    nc_filepath : str or Path
        Directory of intermediate netCDF files.

    Returns
    -------
    dict
        Float data dict with QC masking applied. Keys include PSALs, TEMPs,
        PRESs, JULDs, LONs, LATs, PROFILE_NUMS.
    """
    nc_data = read_intermediate_nc_file(nc_filepath)
    # Apply QC to floats
    nc_data["PSAL_ADJUSTED"][np.where((nc_data["PSAL_ADJUSTED_QC"] == 3) | (nc_data["PSAL_ADJUSTED_QC"] == 4))] = np.nan
    nc_data["PRES_ADJUSTED"][np.where((nc_data["PRES_ADJUSTED_QC"] == 3) | (nc_data["PRES_ADJUSTED_QC"] == 4))] = np.nan
    nc_data["TEMP_ADJUSTED"][np.where((nc_data["TEMP_ADJUSTED_QC"] == 3) | (nc_data["TEMP_ADJUSTED_QC"] == 4))] = np.nan
    
    # Rename keys to be consistent with AXCTD data for easier processing later on
    nc_data["PSALs"] = nc_data.pop("PSAL_ADJUSTED")
    nc_data["PRESs"] = nc_data.pop("PRES_ADJUSTED")
    nc_data["TEMPs"] = nc_data.pop("TEMP_ADJUSTED")

    return nc_data
def filter_float_overlap_date_range(float1_data, float1_name, float2_data, float2_name, just_overlap):
    """
    Filter two float datasets to a common or partially overlapping date range.

    Parameters
    ----------
    float1_data, float2_data : dict
        Float data dicts (must contain JULDs array).
    float1_name, float2_name : str
        Float identifiers (used for print statements).
    just_overlap : bool
        True  = keep only the date range where both floats have profiles.
        False = keep the entire range of whichever float started later.

    Returns
    -------
    tuple of (dict, dict)
        Filtered float1_data and float2_data with out-of-range profiles set to NaN.
    """
    float1_julds = float1_data["JULDs"]
    float2_julds = float2_data["JULDs"]

    # Check to see which float has the later start date
    if float1_julds[0] >= float2_julds[0]:  
        start_date = float1_julds[0]
    else:
        start_date = float2_julds[0]

    if just_overlap == True:
        # Check to see which float has the earlier end date
        if float1_julds[-1] <= float2_julds[-1]:
            end_date = float1_julds[-1] 
        else:
            end_date = float2_julds[-1]
    else:
        # Just take the later end date
        if float1_julds[-1] >= float2_julds[-1]:
            end_date = float1_julds[-1] 
        else:
            end_date = float2_julds[-1]

    print(f"{float1_name} start date: {from_julian_day(float1_julds[0])}, end date: {from_julian_day(float1_julds[-1])}")
    print(f"{float2_name} start date: {from_julian_day(float2_julds[0])}, end date: {from_julian_day(float2_julds[-1])}")
    print(f"Overlap start date: {from_julian_day(start_date)}, end date: {from_julian_day(end_date)}")

    # We want to return a 1D date filter of indexes for each float
    float1_date_filter_1d = (float1_julds >= start_date) & (float1_julds <= end_date)
    float2_date_filter_1d = (float2_julds >= start_date) & (float2_julds <= end_date)

    for var in ["PRESs", "PSALs", "TEMPs"]:
        float1_data[var][~float1_date_filter_1d, :] = np.nan
        float2_data[var][~float2_date_filter_1d, :] = np.nan

    return float1_data, float2_data
def find_psal_at_temp(target_temp, data, show_linear_temp_graph):
    """
    Find practical salinity at a specific target temperature for each profile.

    Uses linear interpolation on TEMP vs PSAL to estimate PSAL at target_temp.
    Profiles where target_temp is outside the observed temperature range return NaN.

    Parameters
    ----------
    target_temp : float
        Target isotherm temperature (°C).
    data : dict
        Float or reference data dict with keys TEMPs, PSALs, PRESs.
    show_linear_temp_graph : bool
        If True, displays a graph of each profile's interpolated PSAL at the
        target temperature (for diagnostic use).

    Returns
    -------
    psal_at_temp : ndarray (n_profiles,)
        Practical salinity at target_temp for each profile. NaN if not found.
    """
    # Assumes we already filtered PRES levels - non-valid are marked as np.nan
    KEY_NAMES = ["PRESs", "TEMPs", "PSALs"]
    local_data = data.copy()

    linear_temp = []
    psal_at_temp = []
    psals = []
    pres = []

    # Linear interpolate TEMP 
    # Input is not 2D, add dummy axis
    if len(local_data[KEY_NAMES[1]].shape) != 2:
        for key in KEY_NAMES:
            local_data[key] = np.expand_dims(local_data[key], axis=0)

    for i in np.arange(local_data[KEY_NAMES[1]].shape[0]):
        # Get rid of any negative temps first to avoid sorting issues
        if np.any(local_data[KEY_NAMES[1]][i] < -2):
            neg_temp_indices = np.where(local_data[KEY_NAMES[1]][i] < -2)
            local_data[KEY_NAMES[0]][i][neg_temp_indices] = np.nan
            local_data[KEY_NAMES[1]][i][neg_temp_indices] = np.nan
            local_data[KEY_NAMES[2]][i][neg_temp_indices] = np.nan
        # Sort PRES + PSAL according to TEMP
        sorted_indices = np.argsort(local_data[KEY_NAMES[1]][i])
        pres_sorted = local_data[KEY_NAMES[0]][i][sorted_indices]
        temp_sorted = local_data[KEY_NAMES[1]][i][sorted_indices]
        psal_sorted = local_data[KEY_NAMES[2]][i][sorted_indices]

        psals.append(psal_sorted)
        pres.append(pres_sorted)
        linear_temp.append(temp_sorted)
    
    if show_linear_temp_graph == True:
        # Plot linear temp profiles w PSAL for visual check
        for i in range(len(linear_temp)):
            plt.scatter(psals[i], linear_temp[i], s = 1)
        plt.gca().invert_yaxis()
        plt.xlabel("PSAL")
        plt.ylabel("Temperature (°C)")
        plt.title("Linear Interpolated TEMPs v PSAL Profiles")
        plt.grid(visible=True)
        plt.show()

    # Now find PSAL at target TEMP
    for i in np.arange(local_data[KEY_NAMES[2]].shape[0]):
        mask = ~np.isnan(linear_temp[i]) & ~np.isnan(psals[i])
        if np.any(mask):
            psal_at_temp.append(np.interp(target_temp, linear_temp[i][mask], psals[i][mask],
                                          left=np.nan, right=np.nan))            
        else:
            psal_at_temp.append(np.nan)

    return np.squeeze(np.array(psal_at_temp))
def make_TS_plot(list_of_data, list_of_labels):
    """
    Create a Temperature-Salinity diagram with sigma-t density contours.

    Plots multiple datasets on the same axes using different colors from the
    'tab10' colormap. Density contours are computed from GSW rho at surface
    pressure and labeled as sigma-t (rho - 1000).

    Function can handle up to 10 different data sources. 

    Parameters
    ----------
    list_of_data : list of dict
        Each dict must have keys PSALs and TEMPs (2D arrays, n_profiles×n_levels).
    list_of_labels : list of str
        Legend labels corresponding to each dataset in list_of_data.
    """
    # Find the min/max of TEMP and PSAL across all datasets to set the limits of the grid
    smin, smax, tmin, tmax = np.inf, -np.inf, np.inf, -np.inf
    for data in list_of_data:
        if smin > np.nanmin(data["PSALs"]):
            smin = np.nanmin(data["PSALs"])
        if smax < np.nanmax(data["PSALs"]):
            smax = np.nanmax(data["PSALs"])
        if tmin > np.nanmin(data["TEMPs"]):
            tmin = np.nanmin(data["TEMPs"])
        if tmax < np.nanmax(data["TEMPs"]):
            tmax = np.nanmax(data["TEMPs"])
    smin -= 1
    smax += 1
    tmin -= 1
    tmax += 1
    # Calculate grid cells needed in x and y dimensions
    xdim = int(np.ceil((smax - smin) / 0.1))
    ydim = int(np.ceil((tmax - tmin)))
    dens = np.zeros((ydim, xdim))
    # Create temp and salt vectors for density
    ti = np.linspace(0, ydim, ydim) + tmin
    si = np.linspace(1, xdim, xdim) * 0.1 + smin
    # Fill grid with densities
    for j in range(ydim):
        for i in range(xdim):
            dens[j, i] = gsw.rho(si[i], ti[j], 0)
    # Subtract 1000 to convert to sigma-t
    dens = dens - 1000

    # Assign a distinct color per data source; tab10 covers up to 10 sources
    colors = plt.cm.tab10.colors

    # Create a line segment collection and legend handle for each data source
    line_segments = []
    legend_handles = []
    for idx, (data, label) in enumerate(zip(list_of_data, list_of_labels)):
        color = colors[idx % len(colors)]
        if len(data["PRESs"].shape) == 1:  # 1D - single profile (e.g. AXCTD)
            segments = np.column_stack((data["PSALs"], data["TEMPs"]))
            lc = LineCollection([segments], color=color, linewidth=1)
        else:  # 2D - multiple profiles (e.g. float)
            valid = ~np.isnan(data["JULDs"])
            segments = [np.column_stack((data["PSALs"][i, :], data["TEMPs"][i, :])) for i in range(data["PSALs"].shape[0]) if valid[i]]
            lc = LineCollection(segments, color=color, linewidth=1)
        line_segments.append(lc)
        legend_handles.append(Line2D([0], [0], color=color, linewidth=1, label=label))

    # Generate plot
    fig, ax = plt.subplots(figsize=(10, 6))
    CS = plt.contour(si, ti, dens, linestyles='dashed', colors='k')
    plt.clabel(CS, fontsize=12, inline=1, fmt='%.2f')

    # Add line segments to the plot
    for lc in line_segments:
        ax.add_collection(lc)

    # Set x and y limits, labels, title, and legend
    plt.xlim([smin + 0.75, smax - 0.75])
    plt.ylim([tmin + 0.75, tmax - 0.75])
    plt.xlabel('Salinity (PSU)')
    plt.ylabel('In-Situ Temperature (degC)')
    plt.title(" vs ".join(list_of_labels) + " TS Diagram")
    ax.legend(handles=legend_handles)
    plt.tight_layout()

    plt.grid(True)
    plt.show()
    
######## READING FUNCTIONS ########
def read_AXCTDs(filepath, bin_size):
    """
    Read one AXCTD drop from an .edf file and return bin-averaged data.

    Parses the tab-delimited data section after the 'Field9' header line,
    extracting depth, temperature, and conductivity. Converts depth to pressure
    using GSW. If bin_size > 0, performs pressure bin averaging and recalculates
    practical salinity.

    Parameters
    ----------
    filepath : str or Path
        Path to a single .edf AXCTD file.
    bin_size : float
        Pressure bin width in dbar. Pass 0 for no binning (return raw data).

    Returns
    -------
    dict
        Keys: PRESs, TEMPs, CNDCs, JULDs, LATS, LONS (all 1D arrays for one cast).
    """

    before_data = True
    field9found = False
    after_data = False
    depth, temp, cndc = [], [], []
    juld, lat, lon = None, None, None

    with open(filepath, mode='r') as file:
        for line in file:
            if after_data == True:
                line = line.split('\t')
                if float(line[2]) == -99 or float(line[3]) == -99 or float(line[4]) == -99:
                    print("AXCTD -99 value, skipping row")
                else:
                    depth.append(float(line[2]))
                    temp.append(float(line[3]))
                    cndc.append(float(line[4]))
            if before_data == True:
                if "Date of Launch" in line:
                    juld = line.strip('\n').split(" ")[-1]
                if "Time of Launch" in line:
                    juld = to_julian_day(datetime.strptime(f"{juld} {line.strip('\n').split(" ")[-1]}", "%m/%d/%Y %H:%M:%S"))
                if "Latitude" in line:
                    temp_lat = line.strip('\n').split(" ")
                    lat = dm_to_decimal(float(temp_lat[-2]), float(temp_lat[-1][:-1]), temp_lat[-1][-1])
                if "Longitude" in line:
                    temp_lon = line.strip('\n').split(" ")
                    lon = dm_to_decimal(float(temp_lon[-2]), float(temp_lon[-1][:-1]), temp_lon[-1][-1])

                line = line.split(' ')
                if field9found == True:
                    after_data = True
                    before_data = False
                if "Field9" in line:
                    field9found = True
    
    # First, convert depth to pressure
    pres = gsw.p_from_z(-np.asarray(depth, dtype=np.float64), lat)
    # bin avg data - PRES, TEMP, CNDC
    bin_edges = np.arange(np.nanmin(pres), np.nanmax(pres) + 2, bin_size)
    binned_pres = stats.binned_statistic(pres, pres, 'mean', bins=bin_edges).statistic
    binned_temp = stats.binned_statistic(pres, temp, 'mean', bins=bin_edges).statistic
    binned_cdnc = stats.binned_statistic(pres, cndc, 'mean', bins=bin_edges).statistic
    binned_psal = gsw.SP_from_C(binned_cdnc, binned_temp, binned_pres)

    return {
        "PRESs": np.squeeze(np.array(binned_pres, dtype=np.float64)),
        "TEMPs": np.squeeze(np.array(binned_temp, dtype=np.float64)),
        "CNDCs": np.squeeze(np.array(binned_cdnc, dtype=np.float64)),
        "PSALs": np.squeeze(np.array(binned_psal, dtype=np.float64)),
        "JULDs": np.squeeze(juld),
        "LATS": lat,
        "LONS": lon
    }
def read_float_11678(txt_file):

    # float_data_11678 = Path(r"C:\Users\szswe\Desktop\sal_drift\11678_All-Dives.txt")
    # outfile = Path(r"C:\Users\szswe\Desktop\sal_drift\11678_ascent_cleaned.txt")

    # with open(float_data_11678, "r", encoding="utf-8", errors="replace") as f:
    #     lines = f.readlines()

    # with open(outfile, "w") as out:
    #     dive = None
    #     lat, lon = None, None
    #     fix_time = None
    #     in_ascent = False

    #     for i, line in enumerate(lines):
    #         line = line.strip()

    #         # Dive number
    #         if line.startswith("Float") and "Dive" in line:
    #             dive = line.split("Dive")[-1].strip()
    #             lat = lon = fix_time = None
    #             in_ascent = False

    #         # DiveStart block: lat / lon / fix time
    #         if line.startswith("Fix kind:") and "DiveStart" in line:
    #             j = i
    #             while j < len(lines):
    #                 l = lines[j].strip()
    #                 if l.startswith("Fix time:"):
    #                     fix_time = l.split(":", 1)[1].strip()
    #                 if l.startswith("Latitude:"):
    #                     lat = l.split(":", 1)[1].strip()
    #                 elif l.startswith("Longitude:"):
    #                     lon = l.split(":", 1)[1].strip()
    #                     break
    #                 j += 1

    #         # Ascent data section
    #         if line == "Ascent Data":
    #             out.write(f"Dive: {dive}\n")
    #             out.write(f"Latitude: {lat}\n")
    #             out.write(f"Longitude: {lon}\n")
    #             out.write(f"Fix time: {fix_time}\n")
    #             out.write("PRES, TEMP, PSAL\n")
    #             in_ascent = True
    #             continue

    #         # End of ascent table
    #         if in_ascent and ("╚" in line or line == ""):
    #             out.write("\n")
    #             in_ascent = False
    #             continue

    #         # Ascent data rows
    #         if in_ascent and line[:1].isdigit():
    #             parts = line.replace("║", "").split()
    #             if len(parts) >= 8:
    #                 pres = parts[1]
    #                 temp = parts[3]
    #                 psal = parts[7]
    #                 out.write(f"{pres}, {temp}, {psal}\n")

    # raise Exception
    with open(txt_file, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    
    profile_nums = []
    lats = []
    lons = []
    pres, temp, psal = [], [], []
    in_data = False
    temp_pres, temp_temp, temp_psal = [], [], []
    juld = []
    first_prof = True

    for line in lines:
        line = line.split()
        if line == []:
            continue

        if line[0] == "Dive:":
            if first_prof == False:
                pres.append(temp_pres)
                temp.append(temp_temp)
                psal.append(temp_psal)
            if first_prof == True:
                first_prof = False
            in_data = False
            temp_pres, temp_temp, temp_psal = [], [], []
            profile_nums.append(int(line[1]))
        if in_data == True:
            temp_pres.append(float(line[0].split(",")[0]))
            temp_temp.append(float(line[1].split(",")[0]))
            temp_psal.append(float(line[2].split(",")[0]))
        if line[0] == "PRES,":
            in_data = True
        if line[0] == "Latitude:":
            if line[1] == 'None':
                lats.append(np.nan)
            else:
                lats.append(float(line[1]))    
        if line[0] == "Longitude:":
            if line[1] == 'None':
                lons.append(np.nan)
            else:
                lons.append(float(line[1]))
        if line[0] == "Fix":
            # Take timestamp convert to python datetime
            if line[2] == "None":
                juld.append(np.nan)
            else:
                juld.append(to_julian_day(datetime.strptime(f"{line[2]} {line[3]}", "%Y-%m-%d %H:%M:%S")))

   
    pres.append(temp_pres)
    temp.append(temp_temp)
    psal.append(temp_psal)

    #interpolated missing dates
    juld = interpolate_missing_julian_days(juld)
    
    return {
        "PRESs": np.squeeze(np.array(list(itertools.zip_longest(*pres, fillvalue=np.nan)))).T,
        "TEMPs": np.squeeze(np.array(list(itertools.zip_longest(*temp, fillvalue=np.nan)))).T,
        "PSALs": np.squeeze(np.array(list(itertools.zip_longest(*psal, fillvalue=np.nan)))).T,
        "JULDs": np.squeeze(juld),
        "LATS": np.squeeze(lats),
        "LONS": np.squeeze(lons)
    }
def read_regular_file(filepath, date, col_names, bin_size = None, cndc = False, lat = None, lon = None):
    """
    Read a generic CSV/TXT data file into a standard float data dict. 
    Regular files are just data files containing column headers and values, with no metadata or special formatting. 
    This function is meant to be flexible for various reference datasets that may come in a simple tabular format.

    Parameters
    ----------
    filepath : str or Path
        Path to the data file (CSV or whitespace-delimited text).
    date : str
        Date string for this dataset (converted to JULD via to_julian_day).
    col_names : list of str
        Column names for [PRES, TEMP, PSAL, CNDC (optional)].
    bin_size : float or None
        If provided, bin-average data at this pressure interval (dbar).
    cndc : bool
        If True, read and return a CNDC column.
    lat, lon : float or None
        Position of the cast (stored as scalar in returned dict).

    Returns
    -------
    dict
        Keys: PSALs, TEMPs, PRESs, JULDs, LATS, LONS (and CNDCs if cndc=True).
    """
    fp_data = pd.read_csv(filepath)
    data_dict = fp_data.to_dict(orient='list')
    juld = to_julian_day(datetime.strptime(date, "%Y-%m-%d %H:%M:%S"))
    
    ctd_data = {
        "PSALs": np.squeeze(np.asarray(data_dict[col_names[2]], dtype=np.float64)),
        "TEMPs": np.squeeze(np.asarray(data_dict[col_names[1]], dtype=np.float64)),
        "PRESs": np.squeeze(np.asarray(data_dict[col_names[0]], dtype=np.float64)),
        "JULDs": np.asarray(juld, dtype=np.float64),
        "LATS": lat,
        "LONS": lon
    }

    if cndc is True:
        ctd_data["CNDCs"] = np.squeeze(np.asarray(data_dict[col_names[3]], dtype=np.float64))
    if bin_size is None:
        return ctd_data
    
    # bin avg data - PRES, TEMP, PSAL/ CNDC
    bin_edges = np.arange(np.nanmin(ctd_data["PRESs"]), np.nanmax(ctd_data["PRESs"]) + 2, bin_size)
    binned_pres = stats.binned_statistic(ctd_data["PRESs"], ctd_data["PRESs"], 'mean', bins=bin_edges).statistic
    binned_temp = stats.binned_statistic(ctd_data["PRESs"], ctd_data["TEMPs"], 'mean', bins=bin_edges).statistic
    if cndc is True:
        binned_cdnc = stats.binned_statistic(ctd_data["PRESs"], ctd_data["CNDCs"], 'mean', bins=bin_edges).statistic
        binned_cdnc_psal = gsw.SP_from_C(binned_cdnc, binned_temp, binned_pres)
    else:
        binned_cdnc_psal = stats.binned_statistic(ctd_data["PRESs"], ctd_data["PSALs"], 'mean', bins=bin_edges).statistic

    return {
        "PRESs": binned_pres,
        "TEMPs": binned_temp,
        "PSALs": binned_cdnc_psal,
        "JULDs": np.asarray(juld, dtype=np.float64),
        "LATS": lat,
        "LONS": lon
    }
def read_corrected_gem_data():
    """
    Read corrected GEM (bottle-corrected) salinity and temperature data from Excel.

    Organizes data by profile. Each profile entry contains depth-sorted PSAL, TEMP,
    PRES measurements along with JULD, LAT, LON.

    Returns
    -------
    dict of dict
        Outer key: profile identifier string.
        Inner keys: PSALs, TEMPs, PRESs, JULDs, LATS, LONS (1D arrays per profile).
    """

    # read corrected gem data
    fp = Path(r"C:\Users\szswe\Desktop\sal_drift\GEM_Data\corrected_gem_from_linda\GEM2024 AML6 CTD_raw_and_corrected_as_per_salt_bottles .xlsx")
    db_corrected_gem = pd.read_excel(fp, 'GEM 2024 AML6 data ADJUSTED')
    
    # Convert to dictionary
    data_dict = db_corrected_gem.to_dict(orient='list')

    # Organize data
    first_line = True
    last_date = None
    dates = []
    lats = []
    lons = []
    sal = []
    temp = []
    pres = []
    sal_t, temp_t, pres_t = [], [], []
    for i in np.arange(0, len(data_dict["date"])):
        # convert dates to julian days 
        temp_date = data_dict["date"][i].to_pydatetime()
        temp_date = to_julian_day(temp_date)
        
        # date didn't change, same profile 
        if last_date == temp_date:
            sal_t.append(data_dict["Salinity (PSU)  Adjusted, with offsets calculated between salt bottle and original salinity"][i])
            temp_t.append(data_dict["Temperature ©"][i])
            pres_t.append(data_dict["Pressure"][i])
        else:
        # different profile
            # check that it is not the first line  
            if first_line is False:
                sal.append(sal_t)
                temp.append(temp_t)
                pres.append(pres_t)

            dates.append(temp_date)
            lats.append(data_dict["lat"][i])
            lons.append(data_dict["lon"][i])
            sal_t = []
            temp_t = []
            pres_t = []            
            first_line = False
        
        last_date = temp_date

    # get last profile 
    sal.append(sal_t)
    temp.append(temp_t)
    pres.append(pres_t)


    GEM_CORR_DATA = {
        "PSALs": np.squeeze(np.array(list(itertools.zip_longest(*sal, fillvalue=np.nan)), dtype = np.float64).T),
        "TEMPs": np.squeeze(np.array(list(itertools.zip_longest(*temp, fillvalue=np.nan)), dtype = np.float64).T),
        "PRESs": np.squeeze(np.array(list(itertools.zip_longest(*pres, fillvalue=np.nan)), dtype = np.float64).T),
        "LATs": np.squeeze(np.array(lats, dtype = np.float64)),
        "LONs": np.squeeze(np.array(lons, dtype = np.float64)),
        "JULDs": np.squeeze(np.array(dates, dtype = np.float64))
    }
    
    # get rid of -9999
    GEM_CORR_DATA["TEMPs"][np.where(GEM_CORR_DATA["TEMPs"] == -9999)] = np.nan

    return GEM_CORR_DATA
def read_bottle_data(fp):
    """
    Read bottle salinity sample data from an XLS file.

    Each profile's bottle samples provide a sparse set of salinity measurements
    at specific depths, useful as high-accuracy reference points.

    Parameters
    ----------
    fp : str or Path
        Path to the XLS bottle data file.

    Returns
    -------
    dict
        Keys: depths (list), samples (list of salinity values), JULDs (list).
    """

    # read in XLS file
    df = pd.read_excel(fp, skiprows=7, usecols=[1,2,5,6,7])

    # Format the data
    depths = []
    sample1 = []
    sample2 = []
    sample3 = []
    date = []
    first_line = True

    for index, row in df.iterrows():
        row = row.to_dict()
        if int(row["Dybde (m)"]) == 1:
            if first_line is False: 
                depths.append(depths_temp)
                sample1.append(sample1_temp)
                sample2.append(sample2_temp)
                sample3.append(sample3_temp)
                date.append(np.float64((date_temp)))
            # new profile
            depths_temp = []
            sample1_temp = []
            sample2_temp = []
            sample3_temp = []
            date_temp = to_julian_day(row["Dato"].to_pydatetime())
            first_line = False
            
            # append info 
            depths_temp.append(int(row["Dybde (m)"]))
            sample1_temp.append(float(row[" 1. aflæsning"]))
            sample2_temp.append(float(row[" 2. aflæsning"]))
            sample3_temp.append(float(row[" 3. aflæsning"]))

        else:
            if not np.isnan(row[" 1. aflæsning"]):
                depths_temp.append(int(row["Dybde (m)"]))
                sample1_temp.append(float(row[" 1. aflæsning"]))
                sample2_temp.append(float(row[" 2. aflæsning"]))
                sample3_temp.append(float(row[" 3. aflæsning"]))
    
    # append last profile 
    depths.append(depths_temp)
    sample1.append(sample1_temp)
    sample2.append(sample2_temp)
    sample3.append(sample3_temp)
    date.append(np.float64((date_temp)))

    # change GEM depths -> pres
    pres_t = []
    for i in depths:
        pres_t.append(gsw.p_from_z(np.negative(i), -53.56))

    return {
        "pres": np.squeeze(np.array(list(itertools.zip_longest(*pres_t, fillvalue=np.nan))).T),
        "sample1": np.squeeze(np.array(list(itertools.zip_longest(*sample1, fillvalue=np.nan))).T),
        "sample2": np.squeeze(np.array(list(itertools.zip_longest(*sample2, fillvalue=np.nan))).T),
        "sample3": np.squeeze(np.array(list(itertools.zip_longest(*sample3, fillvalue=np.nan))).T),
        "date": np.squeeze(np.asarray(date)),
        # all profiles have same lat/ lon after checking GEM data
        "lats": -53.56,
        "lons": 69.28
    }
def read_lorenze_ctd_data(bin_sizes = None):
    """
    Read Lorenz CTD cast data from .cnv format files.

    Returns data from 2 CTD profiles. If bin_sizes is provided, bin-averages
    the data at the specified pressure intervals.

    Parameters
    ----------
    bin_sizes : list of float or None
        Pressure bin widths (one per profile). If None, returns raw data.

    Returns
    -------
    dict
        Keys: PSALs, TEMPs, PRESs, JULDs, LATS, LONS (2D, 2×n_levels).
    """

    ctd1 = Path(r"C:\Users\szswe\Desktop\sal_drift\Lorenz_CTD\GINR_Disko_CTD_2024_06_TA24023.cnv")
    ctd2 = Path(r"C:\Users\szswe\Desktop\sal_drift\Lorenz_CTD\SA25018.cnv")

    time = [to_julian_day(datetime.strptime("2024/06/11T1620", "%Y/%m/%dT%H%M")),
            to_julian_day(datetime.strptime("2025/06/21T1542", "%Y/%m/%dT%H%M"))]  
    lat = [69.2337, 69.202717]
    lon = [-52.5012, -52.376417]
    temp = []  
    cndc = []  
    pres = [] 
    sal = [] 

    with open(ctd1, 'r') as ctd1:
        data_start = False
        for line in ctd1:
            line = line.split()
            if data_start == True:
                temp.append(line[2])
                cndc.append(line[3])
                # convert depth to pressure
                pres.append(gsw.p_from_z(np.negative(float(line[9])), lon[0]))
                sal.append(line[10])

            if line[0] == "*END*":
                data_start = True
    
    temp_2 = []  
    pres_2 = [] 
    sal_2 = [] 
    with open(ctd2, 'r') as ctd2:
        data_start = False
        for line in ctd2:
            line = line.split()
            if data_start == True:
                temp_2.append(line[1])
                # convert depth to pressure
                pres_2.append(float(line[6]))
                sal_2.append(line[2])

            if line[0] == "*END*":
                data_start = True

    overall_pres = [pres, pres_2]
    overall_temp = [temp, temp_2]  
    overall_sal = [sal, sal_2] 

    CTD_data = {
        "PSALs": np.squeeze(np.array(list(itertools.zip_longest(*overall_sal, fillvalue=np.nan)), dtype = np.float64).T),
        "TEMPs": np.squeeze(np.array(list(itertools.zip_longest(*overall_temp, fillvalue=np.nan)), dtype = np.float64).T),
        "PRESs": np.squeeze(np.array(list(itertools.zip_longest(*overall_pres, fillvalue=np.nan)), dtype = np.float64).T),
        "CNDCs": np.squeeze(np.asarray(cndc, dtype = np.float64)),
        "LATs": np.squeeze(np.asarray(lat, dtype = np.float64)),
        "LONs": np.squeeze(np.asarray(lon, dtype = np.float64)),
        "JULDs": np.squeeze(np.asarray(time, dtype = np.float64))
    }

    if bin_sizes == None:
        # get rid of bad TEMP point

        return CTD_data

    binned_psal = []
    binned_temp = []
    binned_pres = []
    binned_cndc = []
    # bin avg data
    for i in [0, 1]:
        nan_index = np.where(~np.isnan(CTD_data["PRESs"][i]))[0][-1] + 1
        if bin_sizes[i] == 0:
            # if 0 is passed in, don't bin avg and just append original data
            binned_pres.append(CTD_data["PRESs"][i, :nan_index])
            binned_psal.append(CTD_data["PSALs"][i, :nan_index])
            binned_temp.append(CTD_data["TEMPs"][i, :nan_index])
            # only CTD1 (i==0) has CNDC; profile 1 gets NaN-filled
            if i == 0:
                binned_cndc.append(CTD_data["CNDCs"][:nan_index])
            else:
                binned_cndc.append(np.full(nan_index, np.nan))
        else:
            num_of_bins = np.ceil((np.nanmax(CTD_data["PRESs"][i, :nan_index]) - np.nanmin(CTD_data["PRESs"][i, :nan_index])) / bin_sizes[i])
            # bin avg pressure
            pres = stats.binned_statistic(CTD_data["PRESs"][i, :nan_index], CTD_data["PRESs"][i, :nan_index], 'mean', bins=num_of_bins).statistic
            binned_pres.append(pres)
            # bin avg other values according to pressure
            psal = stats.binned_statistic(CTD_data["PRESs"][i, :nan_index], CTD_data["PSALs"][i, :nan_index], 'mean', bins=num_of_bins).statistic
            binned_psal.append(psal)
            temp = stats.binned_statistic(CTD_data["PRESs"][i, :nan_index], CTD_data["TEMPs"][i, :nan_index], 'mean', bins=num_of_bins).statistic
            binned_temp.append(temp)
            # only CTD1 (i==0) has CNDC; profile 1 gets NaN-filled
            if i == 0:
                cndc = stats.binned_statistic(CTD_data["PRESs"][i, :nan_index], CTD_data["CNDCs"], 'mean', bins=num_of_bins).statistic
            else:
                cndc = np.full(int(num_of_bins), np.nan)
            binned_cndc.append(cndc)
    
    # throw away bad point in 2nd profile (first PSAL point)
    binned_psal[1][0] = np.nan
    binned_temp[1][0] = np.nan
    binned_pres[1][0] = np.nan
    
    return{
        "PSALs": np.squeeze(np.array(list(itertools.zip_longest(*binned_psal, fillvalue=np.nan)), dtype = np.float64).T),
        "TEMPs": np.squeeze(np.array(list(itertools.zip_longest(*binned_temp, fillvalue=np.nan)), dtype = np.float64).T),
        "PRESs": np.squeeze(np.array(list(itertools.zip_longest(*binned_pres, fillvalue=np.nan)), dtype = np.float64).T),
        "CNDCs": np.squeeze(np.array(binned_cndc)),
        "LATs": np.squeeze(np.asarray(lat)),
        "LONs": np.squeeze(np.asarray(lon)),
        "JULDs": np.squeeze(np.asarray(time))
    }

######## GRAPH FUNCTIONS  ########
def generate_F9185_F9444_avg_PSAL():
    """
    Compare F9185 and F9444 average salinity vs AXCTD reference data.

    Filters both floats to 500–600 dbar, computes mean PSAL per profile, and
    plots time series. Compares against three collocated AXCTD drops.

    Hardcoded float paths and pressure range. For drift analysis.
    """

    # Read in F9185
    nc_filepath = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F9185\F9185_VI")
    F9185_data = read_float_apply_qc(nc_filepath)
    # Read in F9444
    nc_filepath = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F9444\F9444_VI")
    F9444_data = read_float_apply_qc(nc_filepath)
    
    # Read in need AXCTD data
    ctd_fp = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs_2021_Melville")
    axctd_melville = read_AXCTDs(ctd_fp, 2)
    # This CTD looks to be a bit far from the profile?
    ctd_fp_2 = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs2020")
    F9185_deployment_AXCTD = read_AXCTDs(ctd_fp_2, 2)
    # Read in weird float data
    

    # Get overlapping data
    # F9185_data, F9444_data = filter_float_overlap_date_range(F9185_data, "F9185", F9444_data, "F9444", just_overlap=False)
    
    # Filter PRES levels
    pres_min = 500
    pres_max = 600
    F9185_data = filter_pres_levels(F9185_data, pres_min, pres_max)
    F9444_data = filter_pres_levels(F9444_data, pres_min, pres_max)
    F9185_deployment_AXCTD = filter_pres_levels(F9185_deployment_AXCTD, pres_min, pres_max)
    axctd_melville = filter_pres_levels(axctd_melville, pres_min, pres_max)
    F11678_data = filter_pres_levels(float_data_11678, pres_min, pres_max)
    # find AVG PSAL
    F9185_avg_psal = np.nanmean(F9185_data["PSALs"], axis=1)
    F9444_avg_psal = np.nanmean(F9444_data["PSALs"], axis=1)
    F9185_deployment_AXCTD_avg_psal = np.nanmean(F9185_deployment_AXCTD["PSALs"]) 
    axctd_melville_avg_psal = np.nanmean(axctd_melville["PSALs"])
    F11678_avg_psal = np.nanmean(F11678_data["PSALs"], axis=1)
    
    # Plot
    fig, ax = plt.subplots()

    F9185_data["JULDs"] = np.array([from_julian_day(j) for j in F9185_data["JULDs"]])
    F9444_data["JULDs"] = np.array([from_julian_day(j) for j in F9444_data["JULDs"]])
    plt.scatter(F9185_data["JULDs"], F9185_avg_psal, color = "red")
    plt.plot(F9185_data["JULDs"], F9185_avg_psal, color = "red")
    plt.scatter(F9444_data["JULDs"], F9444_avg_psal, color = "blue")
    plt.plot(F9444_data["JULDs"], F9444_avg_psal, color = "blue")
    
    F11678_data["JULDs"] = np.array([from_julian_day(j) for j in F11678_data["JULDs"]])
    plt.scatter(F11678_data["JULDs"], F11678_avg_psal, color = "orange")
    plt.plot(F11678_data["JULDs"], F11678_avg_psal, color = "orange")
    
    plt.scatter(from_julian_day(F9185_deployment_AXCTD["JULDs"]), F9185_deployment_AXCTD_avg_psal, color = "green")
    # axctd_melville["JULDs"] = np.array([from_julian_day(j) for j in axctd_melville["JULDs"]])
    plt.scatter(from_julian_day(axctd_melville["JULDs"]), axctd_melville_avg_psal, color = "purple")
 
    plt.grid(visible=True)
    plt.xlabel("Date")
    plt.ylabel("PSAL")
    custom_legend = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10)
                ]
    # Add legend to the plot
    ax.legend(
        custom_legend,
        ["F9185", "F9444", "F9185 Deployment AXCTD", "Melville AXCTD"],  # Custom labels
        loc='lower left', title="Data Quality"
    )
    plt.title(f"Avg PSAL F9185 and F9444 depth range {pres_min}-{pres_max}")

    plt.show()
def generate_F9185_F9444_PSAL_at_TEMP():
    """
    Compare F9185 and F9444 PSAL at target isotherm (2°C) vs AXCTD reference.

    Filters 300–900 dbar, finds PSAL at 2°C via interpolation, and plots
    time series alongside AXCTD reference values.

    Hardcoded float paths, pressure range, and target temperature.
    """
    # Read in F9185
    nc_filepath = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F9185\F9185_VI")
    F9185_data = read_float_apply_qc(nc_filepath)
    # Read in F9444
    nc_filepath = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F9444\F9444_VI")
    F9444_data = read_float_apply_qc(nc_filepath)
    
    # Read in need AXCTD data
    ctd_fp = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs_2021_Melville")
    axctd_melville = read_AXCTDs(ctd_fp, 2)
    # This CTD looks to be a bit far from the profile?
    ctd_fp_2 = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs2020")
    F9185_deployment_AXCTD = read_AXCTDs(ctd_fp_2, 2)

    # Get overlapping data
    # F9185_data, F9444_data = filter_float_overlap_date_range(F9185_data, "F9185", F9444_data, "F9444", just_overlap=False)
    
    # Filter PRES levels
    pres_min = 300
    pres_max = 900
    F9185_data = filter_pres_levels(F9185_data, pres_min, pres_max)
    F9444_data = filter_pres_levels(F9444_data, pres_min, pres_max)
    F9185_deployment_AXCTD = filter_pres_levels(F9185_deployment_AXCTD, pres_min, pres_max)
    axctd_melville = filter_pres_levels(axctd_melville, pres_min, pres_max)

    # Linear interpolated temp to find PSAL at specific TEMP
    target_temp = 2
    show_graph = False
    F9185_psal_at_temp = find_psal_at_temp(target_temp, F9185_data, show_linear_temp_graph=show_graph)
    F9444_psal_at_temp = find_psal_at_temp(target_temp, F9444_data, show_linear_temp_graph=show_graph)
    F9185_deployment_AXCTD_psal_at_temp = find_psal_at_temp(target_temp, F9185_deployment_AXCTD, show_linear_temp_graph=show_graph)
    axctd_melville_psal_at_temp = find_psal_at_temp(target_temp, axctd_melville, show_linear_temp_graph=show_graph)
    
    # Plot
    fig, ax = plt.subplots()

    F9185_data["JULDs"] = np.array([from_julian_day(j) for j in F9185_data["JULDs"]])
    F9444_data["JULDs"] = np.array([from_julian_day(j) for j in F9444_data["JULDs"]])
    plt.scatter(F9185_data["JULDs"], F9185_psal_at_temp, color = "red")
    plt.plot(F9185_data["JULDs"], F9185_psal_at_temp, color = "red")
    plt.scatter(F9444_data["JULDs"], F9444_psal_at_temp, color = "blue")
    plt.plot(F9444_data["JULDs"], F9444_psal_at_temp, color = "blue")

    plt.scatter(from_julian_day(F9185_deployment_AXCTD["JULDs"]), F9185_deployment_AXCTD_psal_at_temp, color = "green")
    # axctd_melville["JULDs"] = np.array([from_julian_day(j) for j in axctd_melville["JULDs"]])
    plt.scatter(from_julian_day(axctd_melville["JULDs"]), axctd_melville_psal_at_temp, color = "purple")
 
    plt.grid(visible=True)
    plt.xlabel("Date")
    plt.ylabel(f"PSAL at {target_temp} °C")
    custom_legend = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10)
                ]
    # Add legend to the plot
    ax.legend(
        custom_legend,
        ["F9185", "F9444", "F9185 Deployment AXCTD", "Melville AXCTD"],  # Custom labels
        loc='lower left', title="Data Quality"
    )

    plt.title(f"PSAL at target temp {target_temp}°C F9185 and F9444 depth range {pres_min}-{pres_max}")
    plt.show()
def generate_F10052_avg_PSAL():
    """
    Compare F10052 average salinity vs ORP WOOD CTD, GEM, and Lorenz CTD.

    Filters 150–400 dbar, computes mean PSAL, and plots time series with
    reference data from multiple sources.

    Hardcoded float path and pressure range. For drift analysis.
    """
    # Read in F10052
    nc_filepath = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F10052\F10052_FTR")
    F10052_data = read_float_apply_qc(nc_filepath)
    # Read in CTD data
    # ROWS: Timestamp,Temperature_C,Salinity_PSU,Turbidity_NTU,PAR,Pressure_dbar,N2_1_per_s2
    ORP_WOOD_CTD = read_regular_file(Path(r"C:\Users\szswe\Desktop\sal_drift\ORP_WOOD\CSV_FILES\2025_0712_F10052\060671_20250712_2329DUNDEE_downcast_data.csv"),
                                            "2025-07-12 00:00:00",
                                            ['Pressure_dbar', 'Temperature_C', 'Salinity_PSU'])
    # Read in GEM and bottle data 
    corr_GEM = read_corrected_gem_data()
    # bottle_data = read_bottle_data(Path(r"C:\Users\szswe\Desktop\sal_drift\bottle_data\Saltprøveskema Grønland Disko 25-39868, all 2024.xls"))
    # Read in lorenze CTD data
    lorenze_ctd_data = read_lorenze_ctd_data(bin_sizes=[0, 1])

    # Filter PRES levels
    pres_min = 150
    pres_max = 400
    F10052_data = filter_pres_levels(F10052_data, pres_min, pres_max)
    ORP_WOOD_CTD = filter_pres_levels(ORP_WOOD_CTD, pres_min, pres_max)
    corr_GEM = filter_pres_levels(corr_GEM, pres_min, pres_max)
    lorenze_ctd_data = filter_pres_levels(lorenze_ctd_data, pres_min, pres_max)
    # find AVG PSAL
    F10052_avg_psal = np.nanmean(F10052_data["PSALs"], axis=1)
    ORP_WOOD_CTD_avg_psal = np.nanmean(ORP_WOOD_CTD["PSALs"])
    corr_GEM_avg_psal = np.nanmean(corr_GEM["PSALs"], axis=1)
    lorenze_ctd_avg_psal = np.nanmean(lorenze_ctd_data["PSALs"], axis=1)

    # Plot
    fig, ax = plt.subplots()

    F10052_data["JULDs"] = np.array([from_julian_day(j) for j in F10052_data["JULDs"]])
    corr_GEM["JULDs"] = np.array([from_julian_day(j) for j in corr_GEM["JULDs"]])
    lorenze_ctd_data["JULDs"] = np.array([from_julian_day(j) for j in lorenze_ctd_data["JULDs"]])
    plt.scatter(F10052_data["JULDs"], F10052_avg_psal, color = "red")
    plt.plot(F10052_data["JULDs"], F10052_avg_psal, color = "red")
    plt.scatter(corr_GEM["JULDs"], corr_GEM_avg_psal, color = "blue")
    plt.plot(corr_GEM["JULDs"], corr_GEM_avg_psal, color = "blue")
    
    plt.scatter(lorenze_ctd_data["JULDs"], lorenze_ctd_avg_psal, color = "orange")
    plt.scatter(from_julian_day(ORP_WOOD_CTD["JULDs"]), ORP_WOOD_CTD_avg_psal, color = "purple")
    
    # for i in range(bottle_data["date"].shape[0]): # returns 11
    #     ax.scatter(bottle_data["date"][i], bottle_data["sample1"][:, i], color="red", s=20)
    #     ax.scatter(bottle_data["date"][i], bottle_data["sample2"][:, i], color="red", s=20)
    #     ax.scatter(bottle_data["date"][i], bottle_data["sample3"][:, i], color="red", s=20)

    plt.grid(visible=True)
    plt.xlabel("Date")
    plt.ylabel("PSAL")
    custom_legend = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10)
                ]
    # Add legend to the plot
    ax.legend(
        custom_legend,
        ["F10052", "CORR_GEM", "ORP_WOOD_CTD", "LORENZE_CTD"],  # Custom labels
        loc='lower left', title="Data Quality"
    )
    plt.title(f"Avg PSAL F10052 {pres_min}-{pres_max}")

    plt.show()
def generate_F10052_PSAL_at_TEMP(TT, save_dir):
    """
    Compare F10052 PSAL at target isotherm TT vs multiple reference datasets.

    Filters 150–400 dbar and interpolates PSAL at temperature TT for F10052 and
    each reference dataset. Plots results.

    Parameters
    ----------
    TT : float
        Target isotherm temperature (°C).
    save_dir : str or None
        If provided, saves the figure to this directory. Otherwise shows interactively.
    """
    # Read in F10052
    nc_filepath = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F10052\F10052_FTR")
    F10052_data = read_float_apply_qc(nc_filepath)
    # Read in CTD data
    # ROWS: Timestamp,Temperature_C,Salinity_PSU,Turbidity_NTU,PAR,Pressure_dbar,N2_1_per_s2
    ORP_WOOD_CTD = read_regular_file(Path(r"C:\Users\szswe\Desktop\sal_drift\ORP_WOOD\CSV_FILES\2025_0712_F10052\060671_20250712_2329DUNDEE_downcast_data.csv"),
                                            "2025-07-12 00:00:00",
                                            ['Pressure_dbar', 'Temperature_C', 'Salinity_PSU'])
    # Read in GEM and bottle data 
    corr_GEM = read_corrected_gem_data()
    # bottle_data = read_bottle_data(Path(r"C:\Users\szswe\Desktop\sal_drift\bottle_data\Saltprøveskema Grønland Disko 25-39868, all 2024.xls"))
    lorenze_ctd_data = read_lorenze_ctd_data(bin_sizes=[0, 1])

    # Filter PRES levels
    pres_min = 150
    pres_max = 400
    F10052_data = filter_pres_levels(F10052_data, pres_min, pres_max)
    ORP_WOOD_CTD = filter_pres_levels(ORP_WOOD_CTD, pres_min, pres_max)
    corr_GEM = filter_pres_levels(corr_GEM, pres_min, pres_max)
    lorenze_ctd_data = filter_pres_levels(lorenze_ctd_data, pres_min, pres_max)
    # linear interp temp at psal
    target_temp = TT
    show_graph = False
    F10052_psal_at_temp = find_psal_at_temp(target_temp, F10052_data, show_linear_temp_graph=show_graph)
    ORP_WOOD_CTD_psal_at_temp = find_psal_at_temp(target_temp, ORP_WOOD_CTD, show_linear_temp_graph=show_graph)
    corr_GEM_psal_at_temp = find_psal_at_temp(target_temp, corr_GEM, show_linear_temp_graph=show_graph)
    lorenze_ctd_psal_at_temp = find_psal_at_temp(target_temp, lorenze_ctd_data, show_linear_temp_graph=show_graph)
    # F11678_psal_at_temp = find_psal_at_temp(target_temp, float_data_11678, show_linear_temp_graph=show_graph)

    # Plot
    fig, ax = plt.subplots()

    F10052_data["JULDs"] = np.array([from_julian_day(j) for j in F10052_data["JULDs"]])
    corr_GEM["JULDs"] = np.array([from_julian_day(j) for j in corr_GEM["JULDs"]])
    lorenze_ctd_data["JULDs"] = np.array([from_julian_day(j) for j in lorenze_ctd_data["JULDs"]])
    plt.scatter(F10052_data["JULDs"], F10052_psal_at_temp, color = "red")
    plt.plot(F10052_data["JULDs"], F10052_psal_at_temp, color = "red")
    plt.scatter(corr_GEM["JULDs"], corr_GEM_psal_at_temp, color = "blue")
    plt.plot(corr_GEM["JULDs"], corr_GEM_psal_at_temp, color = "blue")
    # plt.scatter(float_data_11678["JULDs"], F11678_psal_at_temp, color = "green")
    # plt.plot(float_data_11678["JULDs"], F11678_psal_at_temp, color = "green")
    
    plt.scatter(from_julian_day(ORP_WOOD_CTD["JULDs"]), ORP_WOOD_CTD_psal_at_temp, color = "purple")
    plt.scatter(lorenze_ctd_data["JULDs"], lorenze_ctd_psal_at_temp, color = "orange")
    
    # for i in range(bottle_data["date"].shape[0]): # returns 11
    #     ax.scatter(bottle_data["date"][i], bottle_data["sample1"][:, i], color="red", s=20)
    #     ax.scatter(bottle_data["date"][i], bottle_data["sample2"][:, i], color="red", s=20)
    #     ax.scatter(bottle_data["date"][i], bottle_data["sample3"][:, i], color="red", s=20)

    plt.grid(visible=True)
    plt.xlabel("Date")
    plt.ylabel("PSAL")
    custom_legend = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10)
                ]
    # Add legend to the plot
    ax.legend(
        custom_legend,
        ["F10052", "CORR_GEM", "ORP_WOOD_CTD", "LORENZE_CTD", ""],  # Custom labels
        loc='lower left', title="Data Quality"
    )

    plt.title(f"PSAL at {target_temp}°C F10052 {pres_min}-{pres_max}")
    plt.savefig(os.path.join(save_dir, f"F10052_PSAL_at_{target_temp}C.png"))
    plt.close()
# F9444
def F9444_avg_PSAL():
    """
    Analyze F9444 salinity drift via average PSAL vs CTD/AXCTD references.

    Filters 700–800 dbar, plots F9444 average PSAL time series alongside Nicole CTD
    and multiple Melville AXCTD reference points.

    Hardcoded float path and pressure range.
    """
    # Read in F9444
    nc_filepath = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F9444\F9444_VI")
    F9444_data = read_float_apply_qc(nc_filepath)
    # Read in ORP_WOOD -> Nicole CTD data from this summer
    ctd_fp = Path(r"C:\Users\szswe\Desktop\sal_drift\ORP_WOOD\CSV_FILES\2025_07_27_F9444\060671_20250727_1003_downcast_data.csv")
    nicole_summer_ctd = read_regular_file(ctd_fp, "2025-07-27 10:03:00", ['Pressure_dbar', 'Temperature_C', 'Salinity_PSU'])
    # Melville AXCTD data - 2020
    axctd_fp = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\Melville_2020\AXCTD-01 TSK Air-launched 20200911131819_435.edf")
    melville_2020_axctd = read_AXCTDs(axctd_fp, 2)
    # Melville AXCTD data - 2021
    # BEST
    melville_2021_435 = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\Melville_2021\AXCTD-01 TSK Air-launched 20210831162303_435.edf")
    # CLOSE ENOUGH?
    melville_2021_432 = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\Melville_2021\AXCTD-01 TSK Air-launched 20210831161311_432.edf")
    melville_2021_428 = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\Melville_2021\AXCTD-01 TSK Air-launched 20210831170725_428.edf")
    melville_2021_442 = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\Melville_2021\AXCTD-01 TSK Air-launched 20210902140000_442.edf")
    
    melville_2021_435 = read_AXCTDs(melville_2021_435, 2)
    melville_2021_432 = read_AXCTDs(melville_2021_432, 2)
    melville_2021_428 = read_AXCTDs(melville_2021_428, 2)
    melville_2021_442 = read_AXCTDs(melville_2021_442, 2)
    
    # Filter PRES levels
    pres_min = 700
    pres_max = 800
    F9444_data = filter_pres_levels(F9444_data, pres_min, pres_max)
    nicole_summer_ctd = filter_pres_levels(nicole_summer_ctd, pres_min, pres_max)
    melville_2020_axctd = filter_pres_levels(melville_2020_axctd, pres_min, pres_max)
    melville_2021_435 = filter_pres_levels(melville_2021_435, pres_min, pres_max)
    melville_2021_432 = filter_pres_levels(melville_2021_432, pres_min, pres_max)
    melville_2021_428 = filter_pres_levels(melville_2021_428, pres_min, pres_max)
    melville_2021_442 = filter_pres_levels(melville_2021_442, pres_min, pres_max)
    # find AVG TEMP
    F9444_avg_psal = np.nanmean(F9444_data["TEMPs"], axis=1)
    nicole_summer_ctd_avg_psal = np.nanmean(nicole_summer_ctd["TEMPs"])
    melville_2020_avg_psal = np.nanmean(melville_2020_axctd["TEMPs"])
    melville_2021_435_avg_psal = np.nanmean(melville_2021_435["TEMPs"])
    melville_2021_432_avg_psal = np.nanmean(melville_2021_432["TEMPs"])
    melville_2021_428_avg_psal = np.nanmean(melville_2021_428["TEMPs"])
    melville_2021_442_avg_psal = np.nanmean(melville_2021_442["TEMPs"])

    # Plot
    fig, ax = plt.subplots()

    F9444_data["JULDs"] = np.array([from_julian_day(j) for j in F9444_data["JULDs"]])
    plt.scatter(F9444_data["JULDs"], F9444_avg_psal, color = "red")
    plt.plot(F9444_data["JULDs"], F9444_avg_psal, color = "red")
    
    plt.scatter(from_julian_day(nicole_summer_ctd["JULDs"]), nicole_summer_ctd_avg_psal, color = "orange")
    plt.scatter(from_julian_day(melville_2020_axctd["JULDs"]), melville_2020_avg_psal, color = "green")
    plt.scatter(from_julian_day(melville_2021_435["JULDs"]), melville_2021_435_avg_psal, color = "purple")
    plt.scatter(from_julian_day(melville_2021_432["JULDs"]), melville_2021_432_avg_psal, color = "pink")
    plt.scatter(from_julian_day(melville_2021_428["JULDs"]), melville_2021_428_avg_psal, color = "brown")
    plt.scatter(from_julian_day(melville_2021_442["JULDs"]), melville_2021_442_avg_psal, color = "black")
 
    plt.grid(visible=True)
    plt.xlabel("Date")
    plt.ylabel("TEMP")
    custom_legend = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='pink', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='brown', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10)
                ]
    # Add legend to the plot
    ax.legend(
        custom_legend,
        ["F9444", "Nicole Summer CTD", "Melville 2020 AXCTD", 
         "Melville 2021 435 BEST", "Melville 2021 432", "Melville 2021 428", "Melville 2021 442"],  # Custom labels
        loc='lower left', title="Data Quality"
    )
    plt.title(f"Avg TEMP F9444 depth range {pres_min}-{pres_max}")

    plt.show()
def F9444_PSAL_AT_TEMP(TT, save_dir):
    """
    Analyze F9444 PSAL at target isotherm TT vs reference data.

    Filters 400–600 dbar, interpolates PSAL at temperature TT for F9444 and
    reference datasets, then plots.

    Parameters
    ----------
    TT : float
        Target isotherm temperature (°C).
    save_dir : str or None
        If provided, saves the figure. Otherwise shows interactively.
    """
    # Read in F9444
    nc_filepath = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F9444\F9444_VI")
    F9444_data = read_float_apply_qc(nc_filepath)
    # Read in ORP_WOOD -> Nicole CTD data from this summer
    ctd_fp = Path(r"C:\Users\szswe\Desktop\sal_drift\ORP_WOOD\CSV_FILES\2025_07_27_F9444\060671_20250727_1003_downcast_data.csv")
    nicole_summer_ctd = read_regular_file(ctd_fp, "2025-07-27 10:03:00", ['Pressure_dbar', 'Temperature_C', 'Salinity_PSU'])
    # Melville AXCTD data - 2020
    axctd_fp = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\Melville_2020\AXCTD-01 TSK Air-launched 20200911131819_435.edf")
    melville_2020_axctd = read_AXCTDs(axctd_fp, 2)
    # Melville AXCTD data - 2021
    # BEST
    melville_2021_435 = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\Melville_2021\AXCTD-01 TSK Air-launched 20210831162303_435.edf")
    # CLOSE ENOUGH?
    melville_2021_432 = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\Melville_2021\AXCTD-01 TSK Air-launched 20210831161311_432.edf")
    melville_2021_428 = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\Melville_2021\AXCTD-01 TSK Air-launched 20210831170725_428.edf")
    melville_2021_442 = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\Melville_2021\AXCTD-01 TSK Air-launched 20210902140000_442.edf")
    
    melville_2021_435 = read_AXCTDs(melville_2021_435, 2)
    melville_2021_432 = read_AXCTDs(melville_2021_432, 2)
    melville_2021_428 = read_AXCTDs(melville_2021_428, 2)
    melville_2021_442 = read_AXCTDs(melville_2021_442, 2)
    
    # Filter PRES levels
    pres_min = 400
    pres_max = 600
    F9444_data = filter_pres_levels(F9444_data, pres_min, pres_max)
    nicole_summer_ctd = filter_pres_levels(nicole_summer_ctd, pres_min, pres_max)
    melville_2020_axctd = filter_pres_levels(melville_2020_axctd, pres_min, pres_max)
    melville_2021_435 = filter_pres_levels(melville_2021_435, pres_min, pres_max)
    melville_2021_432 = filter_pres_levels(melville_2021_432, pres_min, pres_max)
    melville_2021_428 = filter_pres_levels(melville_2021_428, pres_min, pres_max)
    melville_2021_442 = filter_pres_levels(melville_2021_442, pres_min, pres_max)
   
    # Linear interpolated temp to find PSAL at specific TEMP
    target_temp = TT
    show_graph = False
    F9444_psal_at_temp = find_psal_at_temp(target_temp, F9444_data, show_linear_temp_graph=show_graph)
    nicole_summer_ctd_psal_at_temp = find_psal_at_temp(target_temp, nicole_summer_ctd, show_linear_temp_graph=show_graph)
    melville_2020_axctd_psal_at_temp = find_psal_at_temp(target_temp, melville_2020_axctd, show_linear_temp_graph=show_graph)
    melville_2021_435_psal_at_temp = find_psal_at_temp(target_temp, melville_2021_435, show_linear_temp_graph=show_graph)
    melville_2021_432_psal_at_temp = find_psal_at_temp(target_temp, melville_2021_432, show_linear_temp_graph=show_graph)
    melville_2021_428_psal_at_temp = find_psal_at_temp(target_temp, melville_2021_428, show_linear_temp_graph=show_graph)
    melville_2021_442_psal_at_temp = find_psal_at_temp(target_temp, melville_2021_442, show_linear_temp_graph=show_graph)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    F9444_data["JULDs"] = np.array([from_julian_day(j) for j in F9444_data["JULDs"]])
    plt.scatter(F9444_data["JULDs"], F9444_psal_at_temp, color = "red")
    plt.plot(F9444_data["JULDs"], F9444_psal_at_temp, color = "red")
    
    plt.scatter(from_julian_day(nicole_summer_ctd["JULDs"]), nicole_summer_ctd_psal_at_temp, color = "orange")
    plt.scatter(from_julian_day(melville_2020_axctd["JULDs"]), melville_2020_axctd_psal_at_temp, color = "green")
    plt.scatter(from_julian_day(melville_2021_435["JULDs"]), melville_2021_435_psal_at_temp, color = "purple")
    plt.scatter(from_julian_day(melville_2021_432["JULDs"]), melville_2021_432_psal_at_temp, color = "pink")
    plt.scatter(from_julian_day(melville_2021_428["JULDs"]), melville_2021_428_psal_at_temp, color = "brown")
    plt.scatter(from_julian_day(melville_2021_442["JULDs"]), melville_2021_442_psal_at_temp, color = "black")
 
    plt.grid(visible=True)
    plt.xlabel("Date")
    plt.ylabel(f"PSAL at {target_temp} °C")
    custom_legend = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='pink', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='brown', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10)
                ]
    # Add legend to the plot
    ax.legend(
        custom_legend,
        ["F9444", "Nicole Summer CTD", "Melville 2020 AXCTD", 
         "Melville 2021 435 BEST", "Melville 2021 432", "Melville 2021 428", "Melville 2021 442"],  # Custom labels
        loc='lower left', title="Data Quality"
    )
    plt.title(f"PSAL at {target_temp} °C F9444 depth range {pres_min}-{pres_max}")

    plt.savefig(os.path.join(save_dir, f"F9444_PSAL_at_{target_temp}C.png"))
    plt.close()
def F9444_TS():
    """
    Generate a TS diagram for F9444 alongside F9185 and AXCTD reference data.

    Shows the float's T-S relationship relative to co-located reference profiles.
    Hardcoded float paths.
    """
    # Read in F9444
    nc_filepath = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F9444\F9444_VI")
    F9444_data = read_float_apply_qc(nc_filepath)
    # Read in F9185
    nc_filepath_F9185 = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F9185\F9185_VI")
    F9185_data = read_float_apply_qc(nc_filepath_F9185)
    # Read in ORP_WOOD -> Nicole CTD data from this summer
    ctd_fp = Path(r"C:\Users\szswe\Desktop\sal_drift\ORP_WOOD\CSV_FILES\2025_07_27_F9444\060671_20250727_1003_downcast_data.csv")
    nicole_summer_ctd = read_regular_file(ctd_fp, "2025-07-27 10:03:00", ['Pressure_dbar', 'Temperature_C', 'Salinity_PSU'])
    # Melville AXCTD data - 2020
    axctd_fp = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\Melville_2020\AXCTD-01 TSK Air-launched 20200911131819_435.edf")
    melville_2020_axctd = read_AXCTDs(axctd_fp, 2)
    # Melville AXCTD data - 2021
    # 2021 08 31
    melville_2021_430 = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\Melville_2021\AXCTD-01 TSK Air-launched 20210831160449_430.edf")
    melville_2021_432 = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\Melville_2021\AXCTD-01 TSK Air-launched 20210831161311_432.edf")
    melville_2021_435 = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\Melville_2021\AXCTD-01 TSK Air-launched 20210831162303_435.edf")
    melville_2021_428 = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\Melville_2021\AXCTD-01 TSK Air-launched 20210831170725_428.edf")
    melville_2021_430 = read_AXCTDs(melville_2021_430, 2)
    melville_2021_432 = read_AXCTDs(melville_2021_432, 2)
    melville_2021_435 = read_AXCTDs(melville_2021_435, 2)
    melville_2021_428 = read_AXCTDs(melville_2021_428, 2)
    # 2021 09 01
    melville_2021_452 = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\Melville_2021\AXCTD-01 TSK Air-launched 20210901163044_452.edf")
    melville_2021_450 = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\Melville_2021\AXCTD-01 TSK Air-launched 20210901164325_450.edf")
    melville_2021_452 = read_AXCTDs(melville_2021_452, 2)
    melville_2021_450 = read_AXCTDs(melville_2021_450, 2)
    # 2021 09 02
    melville_2021_442 = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\Melville_2021\AXCTD-01 TSK Air-launched 20210902140000_442.edf")
    melville_2021_445 = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\Melville_2021\AXCTD-01 TSK Air-launched 20210902140707_445.edf")
    melville_2021_442 = read_AXCTDs(melville_2021_442, 2)
    melville_2021_445 = read_AXCTDs(melville_2021_445, 2)

    # Filter F9444 dates to match Nicole's summer CTD
    F9444_data["JULDs"][np.where(np.abs(F9444_data["JULDs"] - nicole_summer_ctd["JULDs"]) > 6)] = np.nan
    for i in np.arange(F9444_data["JULDs"].shape[0]):
        if not np.isnan(F9444_data["JULDs"][i]):
            print(f"Prof number {F9444_data['PROFILE_NUMS'][i]} DT: {from_julian_day(F9444_data['JULDs'][i])}")
    print("Nicole's summer CTD DT: ", from_julian_day(nicole_summer_ctd["JULDs"]))
    
    # Filter F9444 dates to match Melville 2021 AXCTDs
    F9444_data["JULDs"][np.where(np.abs(F9444_data["JULDs"] - melville_2021_452["JULDs"]) > 21)] = np.nan
    for i in np.arange(F9444_data["JULDs"].shape[0]):
        if not np.isnan(F9444_data["JULDs"][i]):
            print(f"Prof number {F9444_data['PROFILE_NUMS'][i]} DT: {from_julian_day(F9444_data['JULDs'][i])}")
    print("Melville 2021 AXCTD DT: ", from_julian_day(melville_2021_452["JULDs"]))

    #  # F9185 - before it went HAYWIRE (before prof 182 F9185)
    # F9185_data["JULDs"][np.where(F9185_data["PROFILE_NUMS"] > 182)] = np.nan
    # juld_date_before_182 = np.nanmax(F9185_data["JULDs"])
    # F9444_data["JULDs"][np.where(F9444_data["JULDs"] > juld_date_before_182)] = np.nan    # Filter F9444 dates to match F9185 before it went haywire
    # # Filter F9185 dates to match with F9444 start date
    # F9185_data["JULDs"][np.where(F9185_data["JULDs"] < np.nanmin(F9444_data["JULDs"]))] = np.nan
    # # Filter prof number ranges
    # F9444_data["JULDs"][(F9444_data["PROFILE_NUMS"] < 46) | (F9444_data["PROFILE_NUMS"] > 48)] = np.nan
    # F9185_data["JULDs"][(F9185_data["PROFILE_NUMS"] < 180) | (F9185_data["PROFILE_NUMS"] > 182)] = np.nan
    # # F9444_data["JULDs"][F9444_data["PROFILE_NUMS"] != 16] = np.nan
    # # F9185_data["JULDs"][F9185_data["PROFILE_NUMS"] != 151] = np.nan
    # for i in np.arange(F9444_data["JULDs"].shape[0]):
    #     if not np.isnan(F9444_data["JULDs"][i]):
    #         print(f"F9444 Prof number {F9444_data['PROFILE_NUMS'][i]} DT: {from_julian_day(F9444_data['JULDs'][i])}")
    # for i in np.arange(F9185_data["JULDs"].shape[0]):
    #     if not np.isnan(F9185_data["JULDs"][i]):
    #         print(f"F9185 Prof number {F9185_data['PROFILE_NUMS'][i]} DT: {from_julian_day(F9185_data['JULDs'][i])}")
    
    #  # F9185 and AXCTDs
    # F9185_data["JULDs"][np.where(np.abs(F9185_data["JULDs"] - melville_2021_452["JULDs"]) > 15)] = np.nan
    # for i in np.arange(F9185_data["JULDs"].shape[0]):
    #     if not np.isnan(F9185_data["JULDs"][i]):
    #         print(f"F9185 Prof number {F9185_data['PROFILE_NUMS'][i]} DT: {from_julian_day(F9185_data['JULDs'][i])}")
    # print("Melville 2021 AXCTD DT: ", from_julian_day(melville_2021_452["JULDs"]))


    F9444_data["PSALs"] = F9444_data["PSALs"] + 0.025
    data = [F9444_data, melville_2021_435, melville_2021_442]
    list_of_labels = ["F9444 + 0.025", "Melville 2021 AXCTD 435", "Melville 2021 AXCTD 442"]
    make_TS_plot(data, list_of_labels)
# F9443
def F9443_avg_PSAL():
    """
    Analyze F9443 salinity drift via average PSAL vs ORP WOOD CTD and AXCTD data.

    Filters 500–600 dbar, plots F9443 average PSAL time series alongside reference data.
    Hardcoded float path and pressure range.
    """
    # Read in F9443
    nc_filepath = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F9443\F9443_VI")
    F9443_data = read_float_apply_qc(nc_filepath)
    # CTDs
    ORP_WOOD_fp = Path(r"C:\Users\szswe\Desktop\sal_drift\ORP_WOOD\CSV_FILES\2025_07_14_F9443\060671_20250714_1029_downcast_data.csv")
    orp_wood = read_regular_file(ORP_WOOD_fp, "2025-07-14 10:29:00", ['Pressure_dbar', 'Temperature_C', 'Salinity_PSU'])
    axctd_306 = read_AXCTDs(Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\F9443\AXCTD-01 TSK Air-launched 20210812132422_306.edf"), 2)
    axctd_310 = read_AXCTDs(Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\F9443\AXCTD-01 TSK Air-launched 20210812132756_310.edf"), 2)
    axctd_309 = read_AXCTDs(Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\F9443\AXCTD-01 TSK Air-launched 20210812134137_309.edf"), 2)
    # read in float 11678 for comparison
    float_11678_fp = Path(r"C:\Users\szswe\Desktop\sal_drift\11678_ascent_cleaned.txt")
    float_data_11678 = read_float_11678(float_11678_fp)
    
    # Filter PRES levels
    pres_min = 500
    pres_max = 600
    F9443_data = filter_pres_levels(F9443_data, pres_min, pres_max)
    orp_wood = filter_pres_levels(orp_wood, pres_min, pres_max)
    axctd_306 = filter_pres_levels(axctd_306, pres_min, pres_max)
    axctd_310 = filter_pres_levels(axctd_310, pres_min, pres_max)
    axctd_309 = filter_pres_levels(axctd_309, pres_min, pres_max)
    float_data_11678 = filter_pres_levels(float_data_11678, pres_min, pres_max)
    # find AVG PSAL
    F9443_avg_psal = np.nanmean(F9443_data["PSALs"], axis=1)
    orp_wood_avg_psal = np.nanmean(orp_wood["PSALs"])
    axctd_306_avg_psal = np.nanmean(axctd_306["PSALs"])
    axctd_310_avg_psal = np.nanmean(axctd_310["PSALs"])
    axctd_309_avg_psal = np.nanmean(axctd_309["PSALs"]) 
    float_data_11678_avg_psal = np.nanmean(float_data_11678["PSALs"], axis=1)

    # Plot
    fig, ax = plt.subplots()

    F9443_data["JULDs"] = np.array([from_julian_day(j) for j in F9443_data["JULDs"]])
    plt.scatter(F9443_data["JULDs"], F9443_avg_psal, color = "red")
    plt.plot(F9443_data["JULDs"], F9443_avg_psal, color = "red")
    plt.scatter(from_julian_day(orp_wood["JULDs"]), orp_wood_avg_psal, color = "orange")
    plt.scatter(from_julian_day(axctd_306["JULDs"]), axctd_306_avg_psal, color = "green")
    plt.scatter(from_julian_day(axctd_310["JULDs"]), axctd_310_avg_psal, color = "purple")
    plt.scatter(from_julian_day(axctd_309["JULDs"]), axctd_309_avg_psal, color = "pink")
    
    float_data_11678["JULDs"] = np.array([from_julian_day(j) for j in float_data_11678["JULDs"]])
    plt.scatter(float_data_11678["JULDs"], float_data_11678_avg_psal, color = "black")
    plt.plot(float_data_11678["JULDs"], float_data_11678_avg_psal, color = "black")
    
    plt.grid(visible=True)
    plt.xlabel("Date")
    plt.ylabel("PSAL")
    custom_legend = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='pink', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10) 
                ]
    # Add legend to the plot
    ax.legend(
        custom_legend,
        ["F9443", "ORP_WOOD", "AXCTD_306", "AXCTD_310", "AXCTD_309", "11678"],  # Custom labels
        loc='lower left', title="Data Quality"
    )
    plt.title(f"Avg PSAL F9443 depth range {pres_min}-{pres_max}")

    plt.show()
def F9443_PSAL_AT_TEMP(TT, save_dir):
    """
    Analyze F9443 PSAL at target isotherm TT vs reference data.

    Filters 500–700 dbar, interpolates PSAL at temperature TT, and plots.

    Parameters
    ----------
    TT : float
        Target isotherm temperature (°C).
    save_dir : str or None
        If provided, saves the figure. Otherwise shows interactively.
    """

    # Read in F9443
    nc_filepath = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F9443\F9443_VI")
    F9443_data = read_float_apply_qc(nc_filepath)
    # CTDs
    ORP_WOOD_fp = Path(r"C:\Users\szswe\Desktop\sal_drift\ORP_WOOD\CSV_FILES\2025_07_14_F9443\060671_20250714_1029_downcast_data.csv")
    orp_wood = read_regular_file(ORP_WOOD_fp, "2025-07-14 10:29:00", ['Pressure_dbar', 'Temperature_C', 'Salinity_PSU'])
    axctd_306 = read_AXCTDs(Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\F9443\AXCTD-01 TSK Air-launched 20210812132422_306.edf"), 2)
    axctd_310 = read_AXCTDs(Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\F9443\AXCTD-01 TSK Air-launched 20210812132756_310.edf"), 2)
    axctd_309 = read_AXCTDs(Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs\F9443\AXCTD-01 TSK Air-launched 20210812134137_309.edf"), 2)
    # read in float 11678 for comparison
    float_11678_fp = Path(r"C:\Users\szswe\Desktop\sal_drift\11678_ascent_cleaned.txt")
    float_data_11678 = read_float_11678(float_11678_fp)
    
    # Filter PRES levels
    pres_min = 500
    pres_max = 700
    F9443_data = filter_pres_levels(F9443_data, pres_min, pres_max)
    orp_wood = filter_pres_levels(orp_wood, pres_min, pres_max)
    axctd_306 = filter_pres_levels(axctd_306, pres_min, pres_max)
    axctd_310 = filter_pres_levels(axctd_310, pres_min, pres_max)
    axctd_309 = filter_pres_levels(axctd_309, pres_min, pres_max)
    float_data_11678 = filter_pres_levels(float_data_11678, pres_min, pres_max)
    # find PSAL at TEMP
    target_temp = TT
    show_graph = False
    F9443_psal_at_temp = find_psal_at_temp(target_temp, F9443_data, show_linear_temp_graph=show_graph)
    nicole_summer_ctd_psal_at_temp = find_psal_at_temp(target_temp, orp_wood, show_linear_temp_graph=show_graph)
    axctd_306_psal_at_temp = find_psal_at_temp(target_temp, axctd_306, show_linear_temp_graph=show_graph)
    axctd_310_psal_at_temp = find_psal_at_temp(target_temp, axctd_310, show_linear_temp_graph=show_graph)
    axctd_309_psal_at_temp = find_psal_at_temp(target_temp, axctd_309, show_linear_temp_graph=show_graph)
    float_11678_psal_at_temp = find_psal_at_temp(target_temp, float_data_11678, show_linear_temp_graph=show_graph)

    # Plot
    fig, ax = plt.subplots()

    F9443_data["JULDs"] = np.array([from_julian_day(j) for j in F9443_data["JULDs"]])
    plt.scatter(F9443_data["JULDs"], F9443_psal_at_temp, color = "red")
    plt.plot(F9443_data["JULDs"], F9443_psal_at_temp, color = "red")
    plt.scatter(from_julian_day(orp_wood["JULDs"]), nicole_summer_ctd_psal_at_temp, color = "orange")
    plt.scatter(from_julian_day(axctd_306["JULDs"]), axctd_306_psal_at_temp, color = "green")
    plt.scatter(from_julian_day(axctd_310["JULDs"]), axctd_310_psal_at_temp, color = "purple")
    plt.scatter(from_julian_day(axctd_309["JULDs"]), axctd_309_psal_at_temp, color = "pink")
    
    float_data_11678["JULDs"] = np.array([from_julian_day(j) for j in float_data_11678["JULDs"]])
    plt.scatter(float_data_11678["JULDs"], float_11678_psal_at_temp, color = "black")
    plt.plot(float_data_11678["JULDs"], float_11678_psal_at_temp, color = "black")
    
    plt.grid(visible=True)
    plt.xlabel("Date")
    plt.ylabel("PSAL")
    custom_legend = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='pink', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10) 
                ]
    # Add legend to the plot
    ax.legend(
        custom_legend,
        ["F9443", "ORP_WOOD", "AXCTD_306", "AXCTD_310", "AXCTD_309", "11678"],  # Custom labels
        loc='lower left', title="Data Quality"
    )
    plt.title(f"Avg PSAL F9443 depth range {pres_min}-{pres_max}")

    plt.savefig(os.path.join(save_dir, f"F9443_PSAL_at_{target_temp}C.png"))
    plt.close()
def F9443_TS():
    """
    Generate a TS diagram for F9443 alongside float F11678 and ORP WOOD CTD.

    Shows the float's T-S relationship relative to co-located reference profiles.
    Hardcoded float paths.
    """
    # Read in F9443
    nc_filepath = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F9443\F9443_VI")
    F9443_data = read_float_apply_qc(nc_filepath)
    # Read in F11678
    float_11678_fp = Path(r"C:\Users\szswe\Desktop\sal_drift\11678_ascent_cleaned.txt")
    float_data_11678 = read_float_11678(float_11678_fp)
    # Read in CTD
    orp_wood_fp = Path(r"C:\Users\szswe\Desktop\sal_drift\ORP_WOOD\CSV_FILES\2025_07_14_F9443\060671_20250714_1029_downcast_data.csv")
    nicole_summer_ctd = read_regular_file(orp_wood_fp, "2025-07-14 10:29:00", ['Pressure_dbar', 'Temperature_C', 'Salinity_PSU'])

    # Filters F9443 to overlap with F11678
    # F9443_data["JULDs"][np.where((F9443_data["PROFILE_NUMS"] < 319) | (F9443_data["PROFILE_NUMS"] > 324))] = np.nan
    # #F9443_data["JULDs"][np.where((F9443_data["PROFILE_NUMS"] != 313))] = np.nan
    # for i in np.arange(F9443_data["JULDs"].shape[0]):
    #     if not np.isnan(F9443_data["JULDs"][i]):
    #         print(f"F9443 Prof number {F9443_data['PROFILE_NUMS'][i]} DT: {from_julian_day(F9443_data['JULDs'][i])}")
    # dt1 = to_julian_day(datetime(2025, 11, 1))
    # dt2 = to_julian_day(datetime(2025, 11, 3))
    # float_data_11678["JULDs"][np.where((float_data_11678["JULDs"] < dt1) | (float_data_11678["JULDs"] > dt2))] = np.nan
    # for i in np.arange(float_data_11678["JULDs"].shape[0]):
    #     if not np.isnan(float_data_11678["JULDs"][i]):
    #         print(f"F11678 DT: {from_julian_day(float_data_11678['JULDs'][i])}")

    # Get overlap bw float and nicole
    F9443_data["JULDs"][np.where(np.abs(F9443_data["JULDs"] - nicole_summer_ctd["JULDs"]) > 7)] = np.nan
    for i in np.arange(F9443_data["JULDs"].shape[0]):
        if not np.isnan(F9443_data["JULDs"][i]):
            print(f"F9443 Prof number {F9443_data['PROFILE_NUMS'][i]} DT: {from_julian_day(F9443_data['JULDs'][i])}")
    print("Nicole's summer CTD DT: ", from_julian_day(nicole_summer_ctd["JULDs"]))
    
    F9443_data["PSALs"] = F9443_data["PSALs"] + 0.025
    data = [F9443_data, nicole_summer_ctd]
    list_of_labels = ["F9443 + 0.025", "Nicole's Summer CTD"]
    make_TS_plot(data, list_of_labels)

if __name__ == '__main__':

    F9443_TS()   
    # F9444_TS()
    # TTs = [1.6, 1.7, 1.8, 1.9,
    #        2, 2.1, 2.2, 2.3, 2.4, 2.5]
    # save_dir = Path(r"C:\Users\szswe\Pictures\Screenshots\F9443\500-700")
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # for i in TTs:
    #     F9443_PSAL_AT_TEMP(i, save_dir)