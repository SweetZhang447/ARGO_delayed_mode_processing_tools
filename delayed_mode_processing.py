import csv
import glob
import os 
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import itertools
import netCDF4 as nc4
import pandas as pd
import copy
from scipy.interpolate import interp1d
from graphs_nc import TS_graph_single_dataset_all_profile, deep_section_var_all, flag_TS_data_graphs, flag_range_data_graphs, flag_point_data_graphs, pres_v_var_all
from tools import from_julian_day, to_julian_day, read_intermediate_nc_file, make_intermediate_nc_file, del_all_nan_slices
import gsw 
from matplotlib.lines import Line2D

def interp_missing_lat_lons(lats, lons, dates):
    """
    Interpolates missing lat/ lon vals

    Args:
        lats (Numpy arr of floats): Lat values
        lons (Numpy arr of floats): Lon values
        dates (Numpy arr of floats): Date values in Julian date format

    Returns:
        Numpy arr: returns numpy arrs of filled in interpolated lat and lon vals
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
    NOT USED
    returns arr of interpolated julian dates.

    Args:
        julian_days (Numpy arr of floats): Date values in Julian date format

    Returns:
        Numpy arr: interpolated julian dates
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
    Checks for missing LAT/LON vals, then interpolated missing vals, setting QC flag for interpolated values to 8.

    Args:
        argo_data (dict): dictionary of ARGO delayed mode profile values

    Returns:
        dict: dictionary of ARGO delayed mode profile values with missing vals filled
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
    Checks for missing values in JULDs. 
    If missing, first see if JULD_LOCATION has associated value, if so fill in JULDs with said value, setting
    QC flag to 5. If not, then interpolated missing value and sets QC flag to 8.

    Args:
        argo_data (dict): dictionary of ARGO delayed mode profile values

    Returns:
        dict: dictionary of ARGO delayed mode profile values with missing vals filled
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
    
    return argo_data

def count_check(argo_data):
    """
    Sets PSAL, CNDC, and TEMP ADJUSTED_QC arrays to 3 where NB_SAMPLE_CTD are > 300 or < 1

    Args:
        argo_data (dict): dictionary of ARGO delayed mode profile values

    Returns:
        dict: dictionary of ARGO delayed mode profile values with {PARAM}_ADJUSTED_QC arrs set
    """
    count_mask = np.logical_or(argo_data["NB_SAMPLE_CTD"] > 100,  np.logical_and(argo_data["NB_SAMPLE_CTD"] < 1, argo_data["NB_SAMPLE_CTD"] != -99))
    argo_data["PSAL_ADJUSTED_QC"][count_mask] = 3
    argo_data["CNDC_ADJUSTED_QC"][count_mask] = 3
    argo_data["TEMP_ADJUSTED_QC"][count_mask] = 3

    return argo_data

# NOTE: Use this for RBR inductive sensors to flag data too close to surface
def pres_depth_check(argo_data):
    """
    Checks if pressure is less than 1, if so fill PSAL and CNDC ADJUSTED_QC with 4 to indicate bad value.

    Args:
        argo_data (dict): dictionary of ARGO delayed mode profile values

    Returns:
        dict: dictionary of ARGO delayed mode profile values with {PARAM}_ADJUSTED_QC arrs set
    """
    pres_mask = np.where(argo_data["PRESs"] < 1)
    argo_data["PSAL_ADJUSTED_QC"][pres_mask] = 4
    argo_data["CNDC_ADJUSTED_QC"][pres_mask] = 4
    argo_data["TEMP_ADJUSTED_QC"][np.where(argo_data["PRESs"] < 0.1)] = 4

    return argo_data

def density_inversion_test(argo_data):

    inversion_flags = []

    for i in np.arange(argo_data["PSALs"].shape[0]):
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

def process_autoset_qc_flags(qc_array, adjusted_qc_array, adjusted_data, label, profile_num, julds, pres_data):
    """
    REDUNDANT/ NOT USED as of 2/18
    """
    if not (np.all(qc_array == 0) or np.all(qc_array == 1)):
        if np.any(qc_array == 3) or np.any(qc_array == 4):
            selected_points = flag_point_data_graphs(adjusted_data, pres_data, label, qc_array, profile_num, julds)
            for j in np.arange(0, len(selected_points)):
                index = selected_points[j]
                # Value is bad, mark in both qc + adjusted_qc
                if index == 4:
                    qc_array[j] = 4
                    adjusted_qc_array[j] = 4
                elif index == 3:                    # probably bad/ good, mark in just adjusted
                    adjusted_qc_array[j] = 3
                elif index == 2:
                    adjusted_qc_array[j] = 2
                else:
                    qc_array[j] = 1
                    adjusted_qc_array[j] = 1
            print("Finish setting QC arrays")

            return True, qc_array, adjusted_qc_array
    return False, qc_array, adjusted_qc_array

def verify_autoset_qc_flags(argo_data):
    """
    Pops up series of graphs if there are preset {PARAM}_QC values so delayed mode operator can verify their correctness.

    Args:
        argo_data (dict): dictionary of ARGO delayed mode profile values

    Returns:
        dict: dictionary of ARGO delayed mode profile values
    """

    for i in np.arange(len(argo_data["PROFILE_NUMS"])):
        sal_checked = False
        temp_checked = False
        pres_checked = False
        # trigger_ts = False

        # Check that QC_FLAG_CHECK has not been set yet
        if argo_data["QC_FLAG_CHECK"][i] == 0:

            # Condition to trigger data snapshot graph
            if ((not (np.all(argo_data["TEMP_QC"][i] == 0) or np.all(argo_data["TEMP_QC"][i] == 1))) and 
                (np.any(argo_data["TEMP_QC"][i] == 3) or np.any(argo_data["TEMP_QC"][i] == 4))) or \
                ((not (np.all(argo_data["PSAL_QC"][i] == 0) or np.all(argo_data["PSAL_QC"][i] == 1))) and 
                (np.any(argo_data["PSAL_QC"][i] == 3) or np.any(argo_data["PSAL_QC"][i] == 4))):
                argo_data = data_snapshot_graph(argo_data, argo_data["PROFILE_NUMS"][i])
                sal_checked = True
                temp_checked = True
                pres_checked = True

            # pres_checked, argo_data["PRES_QC"][i], argo_data["PRES_ADJUSTED_QC"][i] = process_autoset_qc_flags(argo_data["PRES_QC"][i], argo_data["PRES_ADJUSTED_QC"][i], None, "PRES", argo_data["PROFILE_NUMS"][i], argo_data["JULDs"][i], argo_data["PRES_ADJUSTED"][i])
            # temp_checked, argo_data["TEMP_QC"][i], argo_data["TEMP_ADJUSTED_QC"][i] = process_autoset_qc_flags(argo_data["TEMP_QC"][i], argo_data["TEMP_ADJUSTED_QC"][i], argo_data["TEMP_ADJUSTED"][i], "TEMP", argo_data["PROFILE_NUMS"][i], argo_data["JULDs"][i], argo_data["PRES_ADJUSTED"][i])
            # sal_checked, argo_data["PSAL_QC"][i], argo_data["PSAL_ADJUSTED_QC"][i] = process_autoset_qc_flags(argo_data["PSAL_QC"][i], argo_data["PSAL_ADJUSTED_QC"][i], argo_data["PSAL_ADJUSTED"][i], "PSAL", argo_data["PROFILE_NUMS"][i], argo_data["JULDs"][i], argo_data["PRES_ADJUSTED"][i])
            # # Trigger TS data graph if any flag required checking
            # if temp_checked or sal_checked:
            #     trigger_ts = True
            # if trigger_ts == True:
            #     selected_points = flag_TS_data_graphs(argo_data["PSAL_ADJUSTED"][i], argo_data["TEMP_ADJUSTED"][i], argo_data["JULDs"][i], argo_data["LONs"][i], argo_data["LATs"][i], argo_data["PRES_ADJUSTED"][i], argo_data["PROFILE_NUMS"][i], argo_data["TEMP_ADJUSTED_QC"][i], argo_data["PSAL_ADJUSTED_QC"][i])
            #     for j in np.arange(0, len(selected_points)):
            #         index = selected_points[j]
            #         # both points are bad
            #         if index == 4:
            #             argo_data["PSAL_ADJUSTED_QC"][i][j] = 4
            #             argo_data["PSAL_QC"][i][j] = 4
            #             argo_data["TEMP_ADJUSTED_QC"][i][j] = 4
            #             argo_data["TEMP_QC"][i][j] = 4 
            #         # sal is bad
            #         elif index == 3:
            #             argo_data["PSAL_ADJUSTED_QC"][i][j] = 4
            #             argo_data["PSAL_QC"][i][j] = 4
            #             argo_data["TEMP_ADJUSTED_QC"][i][j] = 1
            #             argo_data["TEMP_QC"][i][j] = 1
            #         # temp is bad
            #         elif index == 2:
            #             argo_data["PSAL_ADJUSTED_QC"][i][j] = 1
            #             argo_data["PSAL_QC"][i][j] = 1
            #             argo_data["TEMP_ADJUSTED_QC"][i][j] = 4
            #             argo_data["TEMP_QC"][i][j] = 4
            #         # index is 1, both points are good
            #         else: 
            #             argo_data["PSAL_ADJUSTED_QC"][i][j] = 1
            #             argo_data["PSAL_QC"][i][j] = 1
            #             argo_data["TEMP_ADJUSTED_QC"][i][j] = 1
            #             argo_data["TEMP_QC"][i][j] = 1 
            #     print("Finished setting TEMP_QC and PSAL_QC")

            if sal_checked == True and temp_checked == True and pres_checked == True:
                argo_data["QC_FLAG_CHECK"][i] == 1

    return argo_data

def flag_data_points(argo_data, profile_num, data_type):
    """
    Flag data points in {PARAM}_ADJUSTED_QC arrays. Pops out graph pressure v data_type for users
    to look at data and click through to change QC arr. 

    Args:
        argo_data (dict): dictionary of ARGO delayed mode profile values
        profile_num (int): profile number to look at
        data_type (str): determine which data arr: PRES, PSAL, or TEMP to look at 

    Raises:
        Exception: raise Exception if data_type is not PRES, PSAL or TEMP

    Returns:
        dict: dictionary of ARGO delayed mode profile values
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
        selected_points = flag_point_data_graphs(var_arr, pres_arr, data_type, qc_arr, profile_num, date)
    else:
        raise Exception("Invalid data_type")

    for j in np.arange(0, len(selected_points)):
        index = selected_points[j]
        argo_data[f"{data_type}_ADJUSTED_QC"][i][j] = index

    return argo_data

def flag_range_data(argo_data, profile_num, data_type, data_flag = None):
    """
    Flag data ranges in {PARAM}_ADJUSTED_QC arrays. Pops out graph pressure v data_type for users
    to look at data and click through to change QC arr. 

    Args:
        argo_data (dict): dictionary of ARGO delayed mode profile values
        profile_num (int): profile number to look at
        data_type (str): determine which data arr: PRES, PSAL, or TEMP to look at 
        data_flag (optional, int): if specified, sets QC arrays to data_flag, if not, default sets to 4

    Raises:
        Exception: raise Exception if data_type is not PRES, PSAL or TEMP

    Returns:
        dict: dictionary of ARGO delayed mode profile values
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
    Flag data points in {PARAM}_ADJUSTED_QC arrays. Pops out TS graph for users
    to look at data and click through to change QC arr. 

    Args:
        argo_data (dict): dictionary of ARGO delayed mode profile values
        profile_num (int): profile number to look at
        data_type (str): determine which data arr: PRES, PSAL, or TEMP to look at 

    Raises:
        Exception: raise Exception if data_type is not PRES, PSAL or TEMP

    Returns:
        dict: dictionary of ARGO delayed mode profile values
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

def data_snapshot_graph(argo_data, profile_num):
    """
    Pops out graphs to give user quick look at a profile.

    Args:
        argo_data (dict): dictionary of ARGO delayed mode profile values
        profile_num (int): profile number to look at

    Returns:
        dict: dictionary of ARGO delayed mode profile values
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

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Populate the top-left subplot
    selected_points_temp = flag_point_data_graphs(temp_arr, pres_arr, "TEMP", temp_qc, argo_data["PROFILE_NUMS"][i], juld, ax=axs[0, 0], figure=fig)

    # Populate the top-right subplot
    selected_points_psal = flag_point_data_graphs(sal_arr, pres_arr, "PSAL", psal_qc, argo_data["PROFILE_NUMS"][i], juld, ax=axs[0, 1], figure=fig)

    # Populate the bottom-left subplot
    flag_TS_data_graphs(sal_arr, temp_arr, juld, lon, lat, pres_arr, profile_num, temp_qc, psal_qc, ax=axs[1, 0])

    # Fill bottom-right subplot with text
    timestamp = from_julian_day(float(juld))
    axs[1, 1].text(0.5, 0.7, f'Data Snapshot of Profile: {profile_num}', fontsize=12, ha='center', va='center')
    axs[1, 1].text(0.5, 0.6, f'Datetime of Profile: {timestamp.date()} {timestamp.strftime('%H:%M:%S')}', fontsize=12, ha='center', va='center')
    axs[1, 1].text(0.5, 0.5, f'Lat: {lat:.2f} Lon: {lon:.2f}', fontsize=12, ha='center', va='center')
    axs[1, 1].text(0.5, 0.4, f'Flag QC-point feature enabled for TEMP + PSAL graphs', fontsize=12, ha='center', va='center')
    axs[1, 1].axis('off')
    axs[1,0].grid(True)

    # Add legend elements
    custom_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),    # Both bad
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10),   # Salinity bad
        Line2D([0], [0], marker='o', color='w', markerfacecolor='aqua', markersize=10), # Temperature bad
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10)   # Good data
    ]

    # Add legend with bbox_to_anchor to place it nicely at the bottom
    axs[1, 1].legend(
        custom_legend,
        ["Bad", "Probably Bad", "Probably Good", "Good"],
        loc='lower center',
        bbox_to_anchor=(0.5, 0.1),
        ncol=2,
        title="QC Data Quality Flags",
        frameon=False
    )
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    
    for j in np.arange(0, len(selected_points_temp)):
        index = selected_points_temp[j]
        argo_data[f"TEMP_ADJUSTED_QC"][i][j] = index
    for j in np.arange(0, len(selected_points_psal)):
        index = selected_points_psal[j]
        argo_data[f"PSAL_ADJUSTED_QC"][i][j] = index
    print("Finished setting TEMP_ADJUSTED_QC and PSAL_ADJUSTED_QC")
        
    return argo_data

def graph_pres_v_var_all(argo_data, data_type, use_adjusted, float_num, qc_arr_selection, date_indexes):
    """
    Pops out graph of ALL pressures v data_type

    Args:
        argo_data (dict): dictionary of ARGO delayed mode profile values
        data_type (str): PSAL or TEMP
        use_adjusted (bool): True to use {PARAM}_ADJUSTED arrs, otherwise uses {PARAM} arrs
        float_num (str): Float number
        qc_arr_selection (list, int): list of ints to filter QC arrs 
            ex. [0, 1, 2] means we only want data that has an associated QC flag of 0, 1 or 2
        date_indexes (list, int): list of profile indices that correspond with date filter

    Returns:
        Numpy arr of ints: user clicked profile numbers to look at in more detail
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
    profs = argo_data["PROFILE_NUMS"]

    if date_indexes is not None:
        pres = pres[date_indexes]
        var = var [date_indexes]
        julds = julds[date_indexes]
        profs = profs[date_indexes]
 
    selected_profiles = pres_v_var_all(pres, var, julds, profs, data_type, float_num)

    return selected_profiles

def graph_deep_section_var_all(argo_data, data_type, use_adjusted, float_num, qc_arr_selection, date_indexes):
    """
    Pops out deep section graph of data_type.

    Args:
        argo_data (dict): dictionary of ARGO delayed mode profile values
        data_type (str): PSAL or TEMP
        use_adjusted (bool): True to use {PARAM}_ADJUSTED arrs, otherwise uses {PARAM} arrs
        float_num (str): Float number
        qc_arr_selection (list, int): list of ints to filter QC arrs 
            ex. [0, 1, 2] means we only want data that has an associated QC flag of 0, 1 or 2
        date_indexes (list, int): list of profile indices that correspond with date filter
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

    if date_indexes is not None:
        pres = pres[date_indexes]
        var = var [date_indexes]
        julds = julds[date_indexes]

    deep_section_var_all(pres, julds, var, float_num, data_type)

def graph_TS_all(argo_data, use_adjusted, float_num, qc_arr_selection, date_indexes):
    """
    Pops out TS graph for all data. 

    Args:
        argo_data (dict): dictionary of ARGO delayed mode profile values
        use_adjusted (bool): True to use {PARAM}_ADJUSTED arrs, otherwise uses {PARAM} arrs
        float_num (str): Float number
        qc_arr_selection (list, int): list of ints to filter QC arrs 
            ex. [0, 1, 2] means we only want data that has an associated QC flag of 0, 1 or 2
        date_indexes (list, int): list of profile indices that correspond with date filter

    Returns:
        Numpy arr of ints: user clicked profile numbers to look at in more detail
    """

    pres = argo_data["PRES_ADJUSTED"]
    qc_arr_pres = argo_data["PRES_ADJUSTED_QC"]

    if use_adjusted == True:
        psal = argo_data["PSAL_ADJUSTED"]
        temp = argo_data["TEMP_ADJUSTED"]
        qc_arr_psal = argo_data["PSAL_ADJUSTED_QC"]
        qc_arr_temp = argo_data["TEMP_ADJUSTED_QC"]
    else:
        psal = argo_data["PSAL"]
        temp = argo_data["TEMP"]
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
        
    if date_indexes is not None:
        psal = psal[date_indexes]
        temp = temp[date_indexes]
        pres = pres[date_indexes]
        julds = julds[date_indexes]
        lons = lons[date_indexes]
        lats = lats[date_indexes]
        profs = profs[date_indexes]

    selected_profiles = TS_graph_single_dataset_all_profile(psal, temp, julds, lons, lats, pres, profs, float_num)
    
    return selected_profiles
   
def first_time_run(nc_filepath, dest_filepath, float_num):
    """
    First module to run when doing delayed mode processing. Please see functions and
    associated comments for complete list of all checks done to data.

    Args:
        nc_filepath (str): filepath to netcdf files
        dest_filepath (str): filepath to save netcdf files to after processing
        float_num (str): float number
    """

    # Get dir of generated NETCDF files
    argo_data = read_intermediate_nc_file(nc_filepath)

    # CHECK 0: verify vals in [VAR]_QC arrs
    # NOTE:
    #  refer to argo_quality_control_manual: p.22
    # argo_data = verify_autoset_qc_flags(argo_data)
    
    # CHECK 1: fill-in times and set QC flag to 8
    argo_data = juld_check(argo_data)
    
    # CHECK 2: Interpolate missing lat/lons and set QC flags
    # NOTE: passing in JULDs bc if lat/lon is missing -> JULD_LOCATION is missing
    argo_data = lat_lon_check(argo_data)

    # CHECK 3: Set QC flags where counts are too high/low
    argo_data = count_check(argo_data)

    # Check 4: Set QC flags where PRES < 1m
    #          TODO: DOCUMENT
    argo_data  = pres_depth_check(argo_data)

    # ADD test for TEMP
    argo_data["TEMP_ADJUSTED_QC"][np.where(argo_data["TEMPs"] < -2)] = 4

    # Check 5: Density Inversion Check
    argo_data = density_inversion_test(argo_data)

    # Write results back to NETCDF file
    # make_intermediate_nc_file(argo_data, dest_filepath, float_num)  

def manipulate_data_flags(nc_filepath, dest_filepath, float_num, profile_num):
    """
    Function to look at profile in more detail and flag bad points.

    Args:
        nc_filepath (str): filepath to netcdf files
        dest_filepath (str): filepath to save netcdf files to after processing
        float_num (str): float number
        profile_num (int): profile number to look at
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
    argo_data = flag_TS_data(argo_data, profile_num)
    
    # Write results back to NETCDF file
    make_intermediate_nc_file(argo_data, dest_filepath, float_num, profile_num)  

def generate_dataset_graphs(nc_filepath, dest_filepath, float_num, qc_arr_selection, data_type, use_adjusted, date_filter_start, date_filter_end):
    """
    Pops out graphs to look at complete dataset. 

    Args:
        nc_filepath (str): filepath to netcdf files
        dest_filepath (str): filepath to save netcdf files to after processing
        float_num (str): Float number
        qc_arr_selection (list, int): list of ints to filter QC arrs 
            ex. [0, 1, 2] means we only want data that has an associated QC flag of 0, 1 or 2
        data_type (str): PSAL or TEMP
        use_adjusted (bool): True to use {PARAM}_ADJUSTED arrs, otherwise uses {PARAM} arrs
        
    """
    # Get dir of generated NETCDF files
    argo_data = read_intermediate_nc_file(nc_filepath)

    # Filter date if applicable 
    if date_filter_start is not None and date_filter_end is None:
        # We'll grab data from the start date to end of profile data
        date_filter_start_juld = to_julian_day(datetime.strptime(date_filter_start, "%Y_%m_%d_%H_%M_%S"))
        date_filter_indexes = np.where(argo_data["JULDs"] > date_filter_start_juld)[0]

    elif date_filter_start is not None and date_filter_end is not None:
        # We'll grab data from the start - end date 
        date_filter_start_juld = to_julian_day(datetime.strptime(date_filter_start, "%Y_%m_%d_%H_%M_%S"))
        date_filter_end_juld = to_julian_day(datetime.strptime(date_filter_end, "%Y_%m_%d_%H_%M_%S"))
        date_filter_indexes = np.where(
            (argo_data["JULDs"] > date_filter_start_juld) &
            (argo_data["JULDs"] < date_filter_end_juld))[0]

    elif date_filter_start is None and date_filter_end is not None:
        print("Please specify start date if end date is specified")
        return
    else:
        date_filter_indexes = None
        print("Generating graph of all dates")

    selected_profiles_1 = graph_pres_v_var_all(argo_data, data_type, use_adjusted, float_num, qc_arr_selection, date_filter_indexes)
    selected_profiles_2 = graph_TS_all(argo_data, use_adjusted, float_num, qc_arr_selection, date_filter_indexes)
    graph_deep_section_var_all(argo_data, data_type, use_adjusted, float_num, qc_arr_selection, date_filter_indexes)

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

    #nc_filepath = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\argo_to_nc\\F10051_after_visual_inspection"
    #F9186_after_vi_old    F9186_after_visual_inspection_new
    nc_filepath = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\csv_to_nc\\F9186_visual_inspection"

    float_num = "F9186"
    #dest_filepath = "c:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\argo_to_nc\\F10051_after_visual_inspection"
    # F9186_after_first_time_run_new
    dest_filepath = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\csv_to_nc\\F9186_visual_inspection"
    
    if not os.path.exists(dest_filepath):
        os.mkdir(dest_filepath)

    #first_time_run(nc_filepath, dest_filepath, float_num)

    profile_num = 244
    #manipulate_data_flags(nc_filepath, dest_filepath, float_num, profile_num)

    qc_arr_selection = [0, 1, 2] # only want good/ prob good data 
    data_type = "PSAL"                 # either PSAL or TEMP
    use_adjusted = True                
    # FORMAT: YYYY_MM_DD_HH_MM_SS
    """
    If start date is specified with no end date: filters start date - end of profile data
    If both: start date - end date
    """
    date_filter_start = "2020_09_13_00_00_00" 
    date_filter_end = None
    generate_dataset_graphs(nc_filepath, dest_filepath, float_num, qc_arr_selection, data_type, use_adjusted, date_filter_start, date_filter_end)


if __name__ == '__main__':
 
    main()