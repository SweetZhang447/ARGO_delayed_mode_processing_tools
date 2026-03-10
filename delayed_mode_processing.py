"""
delayed_mode_processing.py — QC processing pipeline and interactive flagging for ARGO float data.

This is the core delayed-mode processing script. It reads intermediate netCDF files
produced by make_origin_nc_files.py, runs automated QC checks, provides interactive
matplotlib-based visualization and flagging tools, and writes the results back to
the intermediate netCDF format.

Three main entry-point functions (called from main):
  first_time_run()          — Run once immediately after make_origin_nc_files.py
  manipulate_data_flags()   — Interactive QC flagging for a single profile
  generate_dataset_graphs() — Dataset-wide graphing and batch flagging workflow

QC flag values: 0=no QC, 1=good, 2=prob good, 3=prob bad, 4=bad, 5=changed, 8=interpolated
Julian days referenced to 1950-01-01 00:00:00 UTC.
"""
import os 
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
from scipy.interpolate import interp1d
from graphs_nc import TS_graph_single_dataset_all_profile, deep_section_var_all, density_inversion_test, flag_TS_data_graphs, flag_range_data_graphs, flag_point_data_graphs, pres_v_var_all
from tools import from_julian_day, to_julian_day, read_intermediate_nc_file, make_intermediate_nc_file, del_all_nan_slices
import gsw 
from matplotlib.lines import Line2D
from pathlib import Path

def interp_missing_lat_lons(lats, lons, dates):
    """
    Interpolate missing (NaN) latitude and longitude values using linear interpolation.

    Uses non-NaN positions as anchor points and extrapolates to fill all NaN values.
    Called by lat_lon_check() after detecting missing position data.

    Parameters
    ----------
    lats, lons : ndarray (n_profiles,)
        Latitude and longitude arrays; NaN where position is missing.
    dates : ndarray (n_profiles,)
        Julian day for each profile, used as the interpolation x-axis.

    Returns
    -------
    lats, lons : ndarray (n_profiles,)
        Arrays with NaN values filled by linear interpolation/extrapolation.
    """

    # Mask where values are not NaN
    mask_lat = ~np.isnan(lats)
    mask_lon = ~np.isnan(lons)

    # Perform linear interpolation only for valid LAT values
    interp_func_lat = interp1d(
        dates[mask_lat], lats[mask_lat], bounds_error=False, fill_value="extrapolate"
    )

    # Perform linear interpolation only for valid LON values
    interp_func_lon = interp1d(
        dates[mask_lon], lons[mask_lon], bounds_error=False, fill_value="extrapolate"
    )

    # Fill NaN values using the interpolation function
    interpolated_values_lat = interp_func_lat(dates)
    interpolated_values_lon = interp_func_lon(dates)

    return interpolated_values_lat, interpolated_values_lon

def interpolate_missing_julian_days(julian_days):
    """
    Interpolate missing (NaN) Julian day values. NOT USED in current pipeline.

    Converts Julian days to Gregorian dates via pandas, interpolates, then converts
    back. NaN values at the edges are dropped.

    Parameters
    ----------
    julian_days : ndarray (n_profiles,)
        Julian day array referenced to 1950-01-01; NaN where missing.

    Returns
    -------
    julian_days : ndarray
        Julian days with NaN values filled by linear interpolation.
    """
    # Convert Julian days to Gregorian dates
    gregorian_dates = [from_julian_day(jd) for jd in julian_days]
    
    # Create a DataFrame with the Gregorian dates
    data = pd.DataFrame({'date': gregorian_dates, 'julian_day': julian_days})
    
    # Set the date as the index
    data.set_index('date', inplace=True)
    
    # Interpolate missing Julian days based on the datetime index
    data['julian_day'] = data['julian_day'].interpolate(method='linear')
    
    # Extract the interpolated Julian days, dropping NaN if there are edges
    interpolated_julian_days = data['julian_day'].dropna().to_numpy()
    
    return interpolated_julian_days

def lat_lon_check(argo_data):
    """
    Check for missing LAT/LON values and fill by interpolation.

    Detects NaN positions, calls interp_missing_lat_lons(), and sets POSITION_QC=8
    for any profiles where position was interpolated.

    Parameters
    ----------
    argo_data : dict
        Intermediate netCDF data dict from read_intermediate_nc_file().

    Returns
    -------
    argo_data : dict
        Same dict with LATs, LONs, and POSITION_QC updated.
    """
    
    location_mask = np.logical_or(np.isnan(argo_data["LATs"]), np.isnan(argo_data["LONs"]))
    # interp LAT/ LON vals
    if len(np.where(location_mask == True)[0]) > 0:
        LATs, LONs = interp_missing_lat_lons(argo_data["LATs"], argo_data["LONs"], argo_data["JULDs"])
        argo_data["LATs"] = LATs
        argo_data["LONs"] = LONs
        argo_data["POSITION_QC"][location_mask] = 8
    
    return argo_data

def juld_check(argo_data):
    """
    Check for missing JULD values and fill from JULD_LOCATION or by interpolation.

    For each profile with a missing JULD:
      - If JULD_LOCATION is available: set JULD = JULD_LOCATION, JULD_QC = 5 (changed).
      - Otherwise: interpolate JULD and set JULD_QC = 8 (interpolated).
    Also fills missing JULD_LOCATION values from JULDs.

    Parameters
    ----------
    argo_data : dict
        Intermediate netCDF data dict from read_intermediate_nc_file().

    Returns
    -------
    argo_data : dict
        Same dict with JULDs, JULD_LOCATIONs, and JULD_QC updated.
    """

    juld_mask = np.isnan(argo_data["JULDs"])
    # Step 1: Fill NaNs in JULDs with known values from JULD_LOCATIONs
    if len(np.where(juld_mask == True)[0]) > 0:
        argo_data["JULDs"][juld_mask] = argo_data["JULD_LOCATIONs"][juld_mask]
        argo_data["JULD_QC"][juld_mask] = 5 # QC flag for changed values
    
    # See if there is any NaNs still left
    interp_mask = np.isnan(argo_data["JULDs"])
    if np.any(interp_mask):
        # Interpolate using surrounding non-NaN JULDs
        valid_indices = np.where(~np.isnan(argo_data["JULDs"]))[0]
        missing_indices = np.where(interp_mask)[0]
        if len(valid_indices) >= 2:  # Need at least two points to interpolate
            argo_data["JULDs"][interp_mask] = np.interp(
                missing_indices,
                valid_indices,
                argo_data["JULDs"][valid_indices]
            )
            argo_data["JULD_QC"][interp_mask] = 8  # QC flag for interpolated values

    # Fill in missing JULD_LOCATION values
    interp_mask = np.isnan(argo_data["JULD_LOCATIONs"])
    if np.any(interp_mask):
        missing_indices = np.where(interp_mask)[0]
        argo_data["JULD_LOCATIONs"][missing_indices] = argo_data["JULDs"][missing_indices]
    
    return argo_data

def count_check(argo_data):
    """
    Flag profiles with suspicious sample counts as probably bad (QC=3).

    Sets PSAL_ADJUSTED_QC and TEMP_ADJUSTED_QC to 3 where NB_SAMPLE_CTD > 100
    or where 0 < NB_SAMPLE_CTD < 1 (but not -99, which indicates missing/N/A).

    Parameters
    ----------
    argo_data : dict
        Intermediate netCDF data dict.

    Returns
    -------
    argo_data : dict
        Same dict with PSAL_ADJUSTED_QC and TEMP_ADJUSTED_QC updated.
    """
    count_mask = np.logical_or(argo_data["NB_SAMPLE_CTD"] > 100,  np.logical_and(argo_data["NB_SAMPLE_CTD"] < 1, argo_data["NB_SAMPLE_CTD"] != -99))
    argo_data["PSAL_ADJUSTED_QC"][count_mask] = 3
    argo_data["TEMP_ADJUSTED_QC"][count_mask] = 3

    return argo_data

# NOTE: Use this for RBR inductive sensors to flag data too close to surface
def pres_depth_check(argo_data):
    """
    Flag near-surface data as bad (QC=4) where sensors are unreliable.

    - Sets PSAL_ADJUSTED_QC = 4 where PRES_ADJUSTED < 1 dbar (RBR inductive cell
      is unreliable near surface).
    - Sets TEMP_ADJUSTED_QC = 4 where PRES_ADJUSTED < 0.1 dbar.

    Parameters
    ----------
    argo_data : dict
        Intermediate netCDF data dict.

    Returns
    -------
    argo_data : dict
        Same dict with PSAL_ADJUSTED_QC and TEMP_ADJUSTED_QC updated.
    """
    pres_mask = np.where(argo_data["PRESs"] < 1)
    argo_data["PSAL_ADJUSTED_QC"][pres_mask] = 4
    argo_data["TEMP_ADJUSTED_QC"][np.where(argo_data["PRESs"] < 0.1)] = 4

    return argo_data

def temp_check(argo_data):
    """
    Flag temperature values below -2°C as bad (QC=4).

    Temperatures below -2°C in the Arctic typically indicate ice contamination
    or sensor errors, not actual water temperature.

    Parameters
    ----------
    argo_data : dict
        Intermediate netCDF data dict.

    Returns
    -------
    argo_data : dict
        Same dict with TEMP_ADJUSTED_QC updated.
    """
    argo_data["TEMP_ADJUSTED_QC"][np.where(argo_data["TEMPs"] < -2)] = 4

    return argo_data

def density_inversion_single_prof(argo_data, prof_num):
    """
    Perform a density inversion test and flag inverted levels as bad (QC=4).

    Converts PSAL to Absolute Salinity and TEMP to Conservative Temperature using
    TEOS-10 (GSW), computes in-situ density, and detects levels where density
    decreases with depth. Sets PSAL_ADJUSTED_QC and TEMP_ADJUSTED_QC to 4 for
    both levels involved in each inversion.

    NOTE: not used since first_time_run will already runs density inversion test

    Parameters
    ----------
    argo_data : dict
        Intermediate netCDF data dict.
    prof_num : int
        Profile number to test.

    Returns
    -------
    argo_data : dict
        Same dict with QC arrays updated for inverted levels.
    """

    i =  np.where(argo_data["PROFILE_NUMS"] == prof_num)[0][0]

    inversion_flags = []

    nan_index = np.where(~np.isnan(argo_data["PRESs"][i, :]))[0][-1] + 1
    psal = argo_data["PSALs"][i]
    temp = argo_data["TEMPs"][i]
    pres = argo_data["PRESs"][i]
    lat = argo_data["LATs"][i]
    lon = argo_data["LONs"][i]

    # Convert PSAL to Absolute Salinity
    abs_sal = gsw.SA_from_SP(psal, pres, lon, lat)
    # Convert TEMP to Conservative Temperature
    cons_temp = gsw.CT_from_t(abs_sal, temp, pres)
    # Calculate in-situ density
    dens = gsw.rho(abs_sal, cons_temp, pres)

    if argo_data["PROFILE_NUMS"][i] == 3:
        fig, ax = plt.subplots()
        
        plt.plot(np.diff(dens[:nan_index]), pres[1:nan_index])
        plt.show()

        data_snapshot_graph(argo_data, 3)

    # Check if pressure is strictly increasing
    # NOTE: our data should be all strictly increasing from make_nc_origin files
    # TODO: show Josh
    if np.all(np.diff(pres[:nan_index]) > 0) == True:
        # if it is then calculate difference
        delta_density = np.diff(dens)
        inversion = delta_density < 0
        inversion_flags.append(inversion)
    # Check if pressure is strictly decreasing 
    elif np.all(np.diff(pres[:nan_index]) < 0) == True:
        delta_density = np.diff(dens)
        inversion = delta_density > 0
        inversion_flags.append(inversion)
    else:
        # need to sort pressure so it is strictly increasing
        sorted_indices = np.argsort(pres[:nan_index])
        # apply sorted pressure order to dens array
        sorted_dens = dens[sorted_indices]

        delta_density = np.diff(sorted_dens)
        inversion = delta_density < 0
        inversion_flags.append(inversion)

    # apply mask to arrays
    for i in range(len(inversion_flags)):
        flag = inversion_flags[i]  # boolean array of length N-1
        # Shifted index since np.diff reduces array length by 1
        flagged_indices = np.where(flag)[0] + 1  # flags apply to the deeper value in the pair
        argo_data["PSAL_ADJUSTED_QC"][i][flagged_indices] = 4
        argo_data["TEMP_ADJUSTED_QC"][i][flagged_indices] = 4

    return argo_data

def save_datasnapshot_graphs(nc_filepath, save_dir):
    """
    Save data snapshot PNG graphs for all profiles in a directory.

    Reads intermediate netCDF files, then calls data_snapshot_graph() for each
    profile with save_dir set, which saves a PNG and closes the figure.

    Parameters
    ----------
    nc_filepath : str
        Directory of intermediate netCDF files.
    save_dir : str
        Directory where PNG files are saved.
    """
    # Get dir of generated NETCDF files
    argo_data = read_intermediate_nc_file(nc_filepath)

    for profile_num in argo_data["PROFILE_NUMS"]:
        data_snapshot_graph(argo_data, profile_num, save_dir)
    
def verify_autoset_qc_flags_and_density_inversions(argo_data):
    """
    Verify pre-existing QC flags and density inversions via interactive graphs.

    For each profile, runs density_inversion_test(). Pops up a data_snapshot_graph()
    if the profile has any levels with TEMP_QC or PSAL_QC set to 3 or 4 (from
    real-time ARGO files), or if any density inversions were found.

    This is called during first_time_run() so the analyst can review and accept or
    correct the autoset flags before proceeding.

    Parameters
    ----------
    argo_data : dict
        Intermediate netCDF data dict.

    Returns
    -------
    argo_data : dict
        Same dict, potentially with QC flags modified by the user.
    """

    for i in np.arange(len(argo_data["PROFILE_NUMS"])):

        # Run density inversion test
        failed_idxs = density_inversion_test(argo_data, argo_data["PROFILE_NUMS"][i])

        # Condition to trigger data snapshot graph
        if ((not (np.all(argo_data["TEMP_QC"][i] == 0) or np.all(argo_data["TEMP_QC"][i] == 1))) and 
            (np.any(argo_data["TEMP_QC"][i] == 3) or np.any(argo_data["TEMP_QC"][i] == 4))) or \
            ((not (np.all(argo_data["PSAL_QC"][i] == 0) or np.all(argo_data["PSAL_QC"][i] == 1))) and 
            (np.any(argo_data["PSAL_QC"][i] == 3) or np.any(argo_data["PSAL_QC"][i] == 4))) or \
            failed_idxs:
            argo_data = data_snapshot_graph(argo_data, argo_data["PROFILE_NUMS"][i])

    return argo_data

def flag_data_points(argo_data, profile_num, data_type):
    """
    Interactive point-by-point QC flagging for a single profile and variable.

    Calls flag_point_data_graphs() from graphs_nc.py and maps the returned color
    list back to integer QC flags (red=4, orange=3, aqua=2, green=1).

    Parameters
    ----------
    argo_data : dict
        Intermediate netCDF data dict.
    profile_num : int
        Profile number to flag.
    data_type : str
        One of 'PRES', 'PSAL', or 'TEMP'.

    Returns
    -------
    argo_data : dict
        Same dict with {data_type}_ADJUSTED_QC updated.
    """

    i =  np.where(argo_data["PROFILE_NUMS"] == profile_num)[0][0]
    pres_arr = argo_data["PRES_ADJUSTED"][i]
    date = argo_data["JULDs"][i]

    if data_type == "PRES":
        qc_arr = argo_data["PRES_ADJUSTED_QC"][i]
        selected_points = flag_point_data_graphs(None, pres_arr, "PRES", qc_arr, profile_num, date)
    
    elif (data_type == "PSAL") or (data_type == "TEMP"):
        var_arr = argo_data[f"{data_type}_ADJUSTED"][i]
        qc_arr = argo_data[f"{data_type}_ADJUSTED_QC"][i]
        if data_type == "PSAL":
            selected_points = flag_point_data_graphs(var_arr, pres_arr, data_type, qc_arr, profile_num, date, argo_data)
        else:
            selected_points = flag_point_data_graphs(var_arr, pres_arr, data_type, qc_arr, profile_num, date)

    else:
        raise Exception("Invalid data_type")

    for j in np.arange(0, len(selected_points)):
        index = selected_points[j]
        argo_data[f"{data_type}_ADJUSTED_QC"][i][j] = index

    return argo_data

def flag_range_data(argo_data, profile_num, data_type, data_flag = None):
    """
    Interactive range-based QC flagging for a single profile and variable.

    Calls flag_range_data_graphs() from graphs_nc.py and maps the returned color
    list back to integer QC flags. If data_flag is provided, it overrides the
    color-to-flag mapping (all changed points get data_flag).

    Parameters
    ----------
    argo_data : dict
        Intermediate netCDF data dict.
    profile_num : int
        Profile number to flag.
    data_type : str
        One of 'PRES', 'PSAL', or 'TEMP'.
    data_flag : int, optional
        If provided, forces all modified points to this QC value instead of
        using the color-to-flag mapping. Defaults to 4 (bad) if not specified.

    Returns
    -------
    argo_data : dict
        Same dict with {data_type}_ADJUSTED_QC updated.
    """

    i = np.where(argo_data["PROFILE_NUMS"] == profile_num)[0][0]
    pres_arr = argo_data["PRES_ADJUSTED"][i]
    date = argo_data["JULDs"][i]

    if data_type == "PRES":
        qc_arr = argo_data["PRES_ADJUSTED_QC"][i]
        selected_colors = flag_range_data_graphs(None, pres_arr, "PRES", qc_arr, profile_num, date)

    elif (data_type == "PSAL") or (data_type == "TEMP"):
        var_arr = argo_data[f"{data_type}_ADJUSTED"][i]
        qc_arr = argo_data[f"{data_type}_ADJUSTED_QC"][i]
        selected_colors = flag_range_data_graphs(var_arr, pres_arr, data_type, qc_arr, profile_num, date)
    else:
        raise Exception("Invalid data_type")
    
    # for index, (p1, p2) in enumerate(selected_points):
    #     if p2 is not None:
    #         for j in np.arange(p1, p2 + 1):
    #             argo_data[f"{data_type}_ADJUSTED_QC"][i][j] = data_flag
    #     else:
    #         argo_data[f"{data_type}_ADJUSTED_QC"][i][p1] = data_flag
    for j in np.arange(len(selected_colors)):
        color = selected_colors[j]
        if color == 'red':
            argo_data[f"{data_type}_ADJUSTED_QC"][i][j] = 4
        elif color == 'orange':
            argo_data[f"{data_type}_ADJUSTED_QC"][i][j] = 3
        elif color == 'aqua':
            argo_data[f"{data_type}_ADJUSTED_QC"][i][j] = 2
        elif color == 'green':
            argo_data[f"{data_type}_ADJUSTED_QC"][i][j] = 1

    return argo_data

def flag_TS_data(argo_data, profile_num):
    """
    Interactive TS-diagram QC flagging for a single profile.

    Calls flag_TS_data_graphs() from graphs_nc.py. Maps the returned combined QC
    state (1=both good, 2=temp bad, 3=sal bad, 4=both bad) to individual
    TEMP_ADJUSTED_QC and PSAL_ADJUSTED_QC arrays.

    Parameters
    ----------
    argo_data : dict
        Intermediate netCDF data dict.
    profile_num : int
        Profile number to flag.

    Returns
    -------
    argo_data : dict
        Same dict with PSAL_ADJUSTED_QC and TEMP_ADJUSTED_QC updated.
    """

    i = np.where(argo_data["PROFILE_NUMS"] == profile_num)[0][0]
    sal_arr = argo_data["PSAL_ADJUSTED"][i]
    temp_arr = argo_data["TEMP_ADJUSTED"][i]
    pres_arr = argo_data["PRES_ADJUSTED"][i]
    temp_qc = argo_data["TEMP_ADJUSTED_QC"][i]
    psal_qc = argo_data["PSAL_ADJUSTED_QC"][i]
    lon = argo_data["LONs"][i]
    lat = argo_data["LATs"][i]
    juld = argo_data["JULDs"][i]

    selected_points = flag_TS_data_graphs(sal_arr, temp_arr, juld, lon, lat, pres_arr, profile_num, temp_qc, psal_qc)

    for j in np.arange(0, len(selected_points)):
        index = selected_points[j]
        # both points are bad
        if index == 4:
            argo_data["PSAL_ADJUSTED_QC"][i][j] = 4
            argo_data["TEMP_ADJUSTED_QC"][i][j] = 4
        # sal is bad
        elif index == 3:
            argo_data["PSAL_ADJUSTED_QC"][i][j] = 4
            argo_data["TEMP_ADJUSTED_QC"][i][j] = 1
        # temp is bad
        elif index == 2:
            argo_data["PSAL_ADJUSTED_QC"][i][j] = 1
            argo_data["TEMP_ADJUSTED_QC"][i][j] = 4
        # index is 1, both points are good
        else: 
            argo_data["PSAL_ADJUSTED_QC"][i][j] = 1
            argo_data["TEMP_ADJUSTED_QC"][i][j] = 1
    
    print("Finished setting TEMP_QC and PSAL_QC")
       
    return argo_data

def data_snapshot_graph(argo_data, profile_num, save_dir = None):
    """
    Display (or save) a 2x2 data snapshot for a single profile.

    Layout: TEMP QC flagging (top-left), PSAL QC flagging (top-right),
    TS diagram (bottom-left), profile info text (bottom-right).

    If save_dir is None: displays interactively and returns updated QC arrays.
    If save_dir is provided: saves a PNG to save_dir and closes the figure (no return).

    When saving, points where NB_SAMPLE_CTD > 50 or QC=4 at the profile edges
    are filtered out of the display.

    Parameters
    ----------
    argo_data : dict
        Intermediate netCDF data dict.
    profile_num : int
        Profile number to display.
    save_dir : str, optional
        If provided, saves figure as PNG here instead of showing interactively.

    Returns
    -------
    argo_data : dict
        Updated dict (only when save_dir is None; QC arrays may be modified).
    """

    i = np.where(argo_data["PROFILE_NUMS"] == profile_num)[0][0]
    sal_arr = argo_data["PSAL_ADJUSTED"][i]
    temp_arr = argo_data["TEMP_ADJUSTED"][i]
    pres_arr = argo_data["PRES_ADJUSTED"][i]
    temp_qc = argo_data["TEMP_ADJUSTED_QC"][i]
    psal_qc = argo_data["PSAL_ADJUSTED_QC"][i]
    lon = argo_data["LONs"][i]
    lat = argo_data["LATs"][i]
    juld = argo_data["JULDs"][i]

    juld_qc = None
    # LAT LON interpolated
    if not np.isnan(argo_data["JULD_QC"][i]):
        juld_qc = int(argo_data["JULD_QC"][i])
    else:
        juld_qc = "Real Value"
    
    location_qc= None
    if not np.isnan(argo_data["POSITION_QC"][i]):
        location_qc = argo_data["POSITION_QC"][i]
    else:
        location_qc = "Real Value"

    # For graphs that are being saved to save_dir, get rid of points with weird counts
    if save_dir is not None:
        nb_sample_ctd = argo_data["NB_SAMPLE_CTD"][i]
        valid_count_mask = np.logical_and(nb_sample_ctd <= 150, nb_sample_ctd >= 1)
        sal_arr = np.where(valid_count_mask, sal_arr, np.nan)
        temp_arr = np.where(valid_count_mask, temp_arr, np.nan)
        pres_arr = np.where(valid_count_mask, pres_arr, np.nan)
        # If PSAL_QC/ TEMP_QC arrays are invalid and it's the first/ last point, get rid of it
        for i in range(5):
            if temp_qc[i] in [3.0, 4.0]:
                pres_arr[i] = np.nan
                temp_arr[i] = np.nan
        for i in range(5):
            if psal_qc[i] in [3.0, 4.0]:
                pres_arr[i] = np.nan
                sal_arr[i] = np.nan
        
        # check if there is still valid data
        if len(np.where(~np.isnan(pres_arr))[0]) == 0 or len(np.where(~np.isnan(sal_arr))[0]) == 0 or len(np.where(~np.isnan(temp_arr))[0]) == 0:
            print(f"Skipping profile {profile_num} for data snapshot graph, no valid data")
            return

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Populate the top-left subplot
    selected_points_temp = flag_point_data_graphs(temp_arr, pres_arr, "TEMP", temp_qc, argo_data["PROFILE_NUMS"][i], juld, ax=axs[0, 0], figure=fig, run_inversion=False)

    # Populate the top-right subplot
    selected_points_psal = flag_point_data_graphs(sal_arr, pres_arr, "PSAL", psal_qc, argo_data["PROFILE_NUMS"][i], juld, argodata=argo_data, ax=axs[0, 1], figure=fig, run_inversion=False)

    # Populate the bottom-left subplot
    flag_TS_data_graphs(sal_arr, temp_arr, juld, lon, lat, pres_arr, profile_num, temp_qc, psal_qc, ax=axs[1, 0])

    # Fill bottom-right subplot with text
    timestamp = from_julian_day(float(juld))
    axs[1, 1].text(0.5, 0.7, f'Data Snapshot of Profile: {profile_num}', fontsize=12, ha='center', va='center')
    axs[1, 1].text(0.5, 0.6, f'Datetime of Profile: {timestamp.date()} {timestamp.strftime("%H:%M:%S")} QC: {juld_qc}', fontsize=12, ha='center', va='center')
    axs[1, 1].text(0.5, 0.5, f'Lat: {lat:.2f} Lon: {lon:.2f} QC: {location_qc}', fontsize=12, ha='center', va='center')
    axs[1, 1].text(0.5, 0.4, f'Flag QC-point feature enabled for TEMP + PSAL graphs', fontsize=12, ha='center', va='center')
    axs[1, 1].axis('off')
    axs[1,0].grid(True)

    # Add legend elements
    custom_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),    # Both bad
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10),   # Salinity bad
        Line2D([0], [0], marker='o', color='w', markerfacecolor='aqua', markersize=10), # Temperature bad
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10),   # Good data
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor='fuchsia', markersize=9) 
    ]

    # Add legend with bbox_to_anchor to place it nicely at the bottom
    axs[1, 1].legend(
        custom_legend,
        ["Bad", "Probably Bad", "Probably Good", "Good", "Density Inversion"],
        loc='lower center',
        bbox_to_anchor=(0.5, 0),
        ncol=2,
        title="QC Data Quality Flags",
        frameon=False
    )
    
    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save or show the figure/ do QC
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"profile_{profile_num}_data_snapshot.png"), dpi=300)
        plt.close()
    else:
        plt.show()
        for j in np.arange(0, len(selected_points_temp)):
            index = selected_points_temp[j]
            argo_data[f"TEMP_ADJUSTED_QC"][i][j] = index
        for j in np.arange(0, len(selected_points_psal)):
            index = selected_points_psal[j]
            argo_data[f"PSAL_ADJUSTED_QC"][i][j] = index
        print("Finished setting TEMP_ADJUSTED_QC and PSAL_ADJUSTED_QC")
        
    return argo_data

def graph_pres_v_var_all(argo_data, data_type, use_adjusted, float_num, qc_arr_selection, filter_indexes):
    """
    Display an interactive pressure-vs-variable overview for all (or filtered) profiles.

    Filters data by QC flag selection and optional profile index list, then calls
    pres_v_var_all() from graphs_nc.py. Returns the set of profiles selected by user.

    Parameters
    ----------
    argo_data : dict
        Intermediate netCDF data dict.
    data_type : str
        'PSAL' or 'TEMP'.
    use_adjusted : bool
        If True, use PARAM_ADJUSTED arrays; otherwise use raw PARAM arrays.
    float_num : str
        Float identifier for the plot title.
    qc_arr_selection : list of int
        Only include depth levels with QC values in this list.
    filter_indexes : list of int or None
        If provided, only display these profile indices. None = all profiles.

    Returns
    -------
    selected_profiles : set or ndarray
        Profile numbers clicked/selected by the user.
    """

    assert data_type == "PSAL" or data_type == "TEMP"
   
    pres = argo_data["PRES_ADJUSTED"]
    qc_arr_pres = argo_data["PRES_ADJUSTED_QC"]

    if use_adjusted == True:
        var = argo_data[f"{data_type}_ADJUSTED"]
        qc_arr_var = argo_data[f"{data_type}_ADJUSTED_QC"]
    else:
        var = argo_data[f"{data_type}s"]
        qc_arr_var = argo_data[f"{data_type}_QC"]
    
    # apply qc_arr to data
    mask_var = np.isin(qc_arr_var, qc_arr_selection)
    mask_pres = np.isin(qc_arr_pres, qc_arr_selection)
    pres = np.where(mask_pres, pres, np.nan)
    var = np.where(mask_var, var, np.nan)
    julds = argo_data["JULDs"]
    profs = argo_data["PROFILE_NUMS"]

    if filter_indexes is not None:
        pres = pres[filter_indexes]
        var = var [filter_indexes]
        julds = julds[filter_indexes]
        profs = profs[filter_indexes]
 
    selected_profiles = pres_v_var_all(pres, var, julds, profs, data_type, float_num)

    return selected_profiles

def graph_deep_section_var_all(argo_data, data_type, use_adjusted, float_num, qc_arr_selection, filter_indexes):
    """
    Display a deep-section graph for all (or filtered) profiles.

    Filters data by QC flag selection and optional profile index list, then calls
    deep_section_var_all() from graphs_nc.py.

    Parameters
    ----------
    argo_data : dict
        Intermediate netCDF data dict.
    data_type : str
        'PSAL' or 'TEMP'.
    use_adjusted : bool
        If True, use PARAM_ADJUSTED arrays.
    float_num : str
        Float identifier for the plot title.
    qc_arr_selection : list of int
        Only include depth levels with QC values in this list.
    filter_indexes : list of int or None
        If provided, only display these profile indices. None = all profiles.
    """
    assert data_type == "PSAL" or data_type == "TEMP"

    pres = argo_data["PRES_ADJUSTED"]
    qc_arr_pres = argo_data["PRES_ADJUSTED_QC"]

    if use_adjusted == True:
        var = argo_data[f"{data_type}_ADJUSTED"]
        qc_arr_var = argo_data[f"{data_type}_ADJUSTED_QC"]
    else:
        var = argo_data[f"{data_type}"]
        qc_arr_var = argo_data[f"{data_type}_QC"]
    
    # apply qc_arr to data
    mask_var = np.isin(qc_arr_var, qc_arr_selection)
    mask_pres = np.isin(qc_arr_pres, qc_arr_selection)

    pres = np.where(mask_pres, pres, np.nan)
    var = np.where(mask_var, var, np.nan)

    julds = argo_data["JULDs"]

    if filter_indexes is not None:
        pres = pres[filter_indexes]
        var = var [filter_indexes]
        julds = julds[filter_indexes]

    deep_section_var_all(pres, julds, var, float_num, data_type)

def graph_TS_all(argo_data, use_adjusted, float_num, qc_arr_selection, filter_indexes):
    """
    Display an interactive TS diagram for all (or filtered) profiles.

    Filters data by QC flag selection and optional profile index list, then calls
    TS_graph_single_dataset_all_profile() from graphs_nc.py. Returns selected profiles.

    Parameters
    ----------
    argo_data : dict
        Intermediate netCDF data dict.
    use_adjusted : bool
        If True, use PARAM_ADJUSTED arrays.
    float_num : str
        Float identifier for the plot title.
    qc_arr_selection : list of int
        Only include depth levels with QC values in this list.
    filter_indexes : list of int or None
        If provided, only display these profile indices. None = all profiles.

    Returns
    -------
    selected_profiles : set or ndarray
        Profile numbers clicked/selected by the user.
    """

    pres = argo_data["PRES_ADJUSTED"]
    qc_arr_pres = argo_data["PRES_ADJUSTED_QC"]

    if use_adjusted == True:
        psal = argo_data["PSAL_ADJUSTED"]
        temp = argo_data["TEMP_ADJUSTED"]
        qc_arr_psal = argo_data["PSAL_ADJUSTED_QC"]
        qc_arr_temp = argo_data["TEMP_ADJUSTED_QC"]
    else:
        psal = argo_data["PSALs"]
        temp = argo_data["TEMPs"]
        qc_arr_psal = argo_data["PSAL_QC"]
        qc_arr_temp = argo_data["TEMP_QC"]  

    # apply qc_arr to data
    mask_psal = np.isin(qc_arr_psal, qc_arr_selection)
    mask_pres = np.isin(qc_arr_pres, qc_arr_selection)
    mask_temp = np.isin(qc_arr_temp, qc_arr_selection)

    pres = np.where(mask_pres, pres, np.nan)
    temp = np.where(mask_temp, temp, np.nan)     
    psal = np.where(mask_psal, psal, np.nan)

    julds = argo_data["JULDs"]     
    lons = argo_data["LONs"]
    lats = argo_data["LATs"]
    profs = argo_data["PROFILE_NUMS"]
        
    if filter_indexes is not None:
        psal = psal[filter_indexes]
        temp = temp[filter_indexes]
        pres = pres[filter_indexes]
        julds = julds[filter_indexes]
        lons = lons[filter_indexes]
        lats = lats[filter_indexes]
        profs = profs[filter_indexes]

    selected_profiles = TS_graph_single_dataset_all_profile(psal, temp, julds, lons, lats, pres, profs, float_num)
    
    return selected_profiles
   
def first_time_run(nc_filepath, dest_filepath, float_num):
    """
    Run the full automated QC pipeline on a newly generated set of intermediate files.

    Must be called once after make_origin_nc_files.py has generated the intermediate
    netCDF files. Sequentially runs:
      1. juld_check()
      2. lat_lon_check()
      3. verify_autoset_qc_flags_and_density_inversions()
      4. count_check()
      5. pres_depth_check()
      6. temp_check()

    Then writes results back to disk via make_intermediate_nc_file().

    Parameters
    ----------
    nc_filepath : str
        Input directory of intermediate netCDF files.
    dest_filepath : str
        Output directory where updated files are written.
    float_num : str
        Float identifier (e.g. 'F9186').
    """

    # Get dir of generated NETCDF files
    argo_data = read_intermediate_nc_file(nc_filepath)

    # CHECK 1: fill-in times and set QC flag to 8
    argo_data = juld_check(argo_data)
    
    # CHECK 2: Interpolate missing lat/lons and set QC flags
    # NOTE: passing in JULDs bc if lat/lon is missing -> JULD_LOCATION is missing
    argo_data = lat_lon_check(argo_data)

    # CHECK 3: verify vals in [VAR]_QC arrs
    # NOTE: refer to argo_quality_control_manual: p.22
    argo_data = verify_autoset_qc_flags_and_density_inversions(argo_data)

    # CHECK 4: Set QC flags where counts are too high/low
    argo_data = count_check(argo_data)

    # Check 5: Set QC flags where PRES < 1m, 
    argo_data  = pres_depth_check(argo_data)

    # Check 6: Set TEMP QC flags where TEMP < -2 and where PRES < 0.1
    argo_data  = temp_check(argo_data)

    # Write results back to NETCDF file
    make_intermediate_nc_file(argo_data, dest_filepath, float_num)  

def manipulate_data_flags(nc_filepath, dest_filepath, float_num, profile_num):
    """
    Interactive QC flagging session for a single profile.

    Reads intermediate netCDF files, displays a data_snapshot_graph, then shows
    flag_range_data graphs for PSAL and TEMP. Writes the result back to disk.

    Parameters
    ----------
    nc_filepath : str
        Input directory of intermediate netCDF files.
    dest_filepath : str
        Output directory where updated files are written.
    float_num : str
        Float identifier (e.g. 'F9186').
    profile_num : int
        Profile number to inspect and flag.
    """
    
    # Get dir of generated NETCDF files
    argo_data = read_intermediate_nc_file(nc_filepath)

    argo_data = data_snapshot_graph(argo_data, profile_num)

    # Flag individual data points
    #argo_data = flag_data_points(argo_data, profile_num, "PRES")
    #argo_data = flag_data_points(argo_data, profile_num, "PSAL")
    #argo_data = flag_data_points(argo_data, profile_num, "TEMP")

   
    # Get rid of range of data
    # argo_data = flag_range_data(argo_data, profile_num, "PRES")
    argo_data = flag_range_data(argo_data, profile_num, "PSAL")
    argo_data = flag_range_data(argo_data, profile_num, "TEMP")

    # TS diagram
    #argo_data = flag_TS_data(argo_data, profile_num)

    # Write results back to NETCDF file
    make_intermediate_nc_file(argo_data, dest_filepath, float_num, profile_num)  

def generate_dataset_graphs(nc_filepath, dest_filepath, float_num, qc_arr_selection, data_type, use_adjusted, date_filter_start, date_filter_end, prof_num_filter):
    """
    Dataset-wide interactive graphing and batch flagging workflow.

    Loads all intermediate netCDF files, optionally filters by date range or
    profile number range, then shows:
      1. graph_pres_v_var_all()
      2. graph_TS_all()
      3. graph_deep_section_var_all()

    For each profile selected in steps 1 or 2, pops up a data_snapshot_graph
    and flag_range_data for TEMP and PSAL. Saves updated files to dest_filepath.

    Parameters
    ----------
    nc_filepath : str
        Input directory of intermediate netCDF files.
    dest_filepath : str
        Output directory for updated files.
    float_num : str
        Float identifier (e.g. 'F9186').
    qc_arr_selection : list of int
        QC values to include in graphs (e.g. [0, 1, 2] for good/prob good).
    data_type : str
        'PSAL' or 'TEMP'.
    use_adjusted : bool
        If True, use PARAM_ADJUSTED arrays.
    date_filter_start : str or None
        Start date filter in format 'YYYY_MM_DD_HH_MM_SS'. None = no filter.
    date_filter_end : str or None
        End date filter in format 'YYYY_MM_DD_HH_MM_SS'. None = start to end of data.
    prof_num_filter : str or None
        Profile number range in format 'START-END' (e.g. '5-7'). None = no filter.
        Mutually exclusive with date filters.
    """
    # Get dir of generated NETCDF files
    argo_data = read_intermediate_nc_file(nc_filepath)

    # Filter date if applicable 
    if date_filter_start is not None and date_filter_end is None:
        # We'll grab data from the start date to end of profile data
        date_filter_start_juld = to_julian_day(datetime.strptime(date_filter_start, "%Y_%m_%d_%H_%M_%S"))
        indexes = np.where(argo_data["JULDs"] > date_filter_start_juld)[0]

    elif date_filter_start is not None and date_filter_end is not None:
        # We'll grab data from the start - end date 
        date_filter_start_juld = to_julian_day(datetime.strptime(date_filter_start, "%Y_%m_%d_%H_%M_%S"))
        date_filter_end_juld = to_julian_day(datetime.strptime(date_filter_end, "%Y_%m_%d_%H_%M_%S"))
        indexes = np.where(
            (argo_data["JULDs"] > date_filter_start_juld) &
            (argo_data["JULDs"] < date_filter_end_juld))[0]

    elif date_filter_start is None and date_filter_end is not None:
        print("Please specify start date if end date is specified")
        return
    else:
        if prof_num_filter is not None:
            prof_filter_split = prof_num_filter.split("-")
            indexes = np.where(
                (argo_data["PROFILE_NUMS"] >= int(prof_filter_split[0])) &
                (argo_data["PROFILE_NUMS"] <= int(prof_filter_split[1])))
        else:
            indexes = None
            print("Generating graph of all dates")
        
    selected_profiles_1 = graph_pres_v_var_all(argo_data, data_type, use_adjusted, float_num, qc_arr_selection, indexes)
    selected_profiles_2 = graph_TS_all(argo_data, use_adjusted, float_num, qc_arr_selection, indexes)
    graph_deep_section_var_all(argo_data, data_type, use_adjusted, float_num, qc_arr_selection, indexes)

    # determine selected profiles based on returns of previous functions
    if selected_profiles_1 is not None and selected_profiles_2 is not None:
        selected_profiles = selected_profiles_1.union(selected_profiles_2)
    elif selected_profiles_1 is not None:
        selected_profiles = selected_profiles_1
    elif selected_profiles_2 is not None:
        selected_profiles = selected_profiles_2
    else:
        selected_profiles = None

    if selected_profiles is not None:
        for prof_num in selected_profiles:
            argo_data = data_snapshot_graph(argo_data, prof_num)
            argo_data = flag_range_data(argo_data, prof_num, "PSAL")
            argo_data = flag_range_data(argo_data, prof_num, "TEMP")
            make_intermediate_nc_file(argo_data, dest_filepath, float_num, prof_num)  


def main():
    # float_num = "7902322"
    # nc_filepath = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F9443\F9443_new_0")
    # dest_filepath = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F9443\ftrnew")

    # if not os.path.exists(dest_filepath):
    #     os.mkdir(dest_filepath)

    nc_filepath = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F10052\F10052_FTR")
    save_dir = Path(r"C:\Users\szswe\Desktop\kml_script\FLOAT_KML_FILES\F10052\images")
    save_datasnapshot_graphs(nc_filepath, save_dir)
    
    # first_time_run(nc_filepath, dest_filepath, float_num)
    raise Exception
    profile_num = 198
    # manipulate_data_flags(nc_filepath, dest_filepath, float_num, profile_num)
    # raise Exception
    qc_arr_selection = [0, 1, 2] # only want good/ prob good data 
    data_type = "TEMP"                 # either PSAL or TEMP
    use_adjusted = True    
    prof_num_filter = None
    # FORMAT: YYYY_MM_DD_HH_MM_SS
    # If start date is specified with no end date: filters start date - end of profile data
    # If both: start date - end date
    # date_filter_start = "2024_01_01_00_00_00"
    # date_filter_end = "2024_06_01_00_00_00"
    date_filter_start = None 
    date_filter_end = None
    generate_dataset_graphs(nc_filepath, dest_filepath, float_num, qc_arr_selection, data_type, use_adjusted, date_filter_start, date_filter_end, prof_num_filter)

if __name__ == '__main__':
 
    main()