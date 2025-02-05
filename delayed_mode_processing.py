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

def interp_missing_lat_lons(lats, lons, dates):

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
    
    location_mask = np.logical_or(np.isnan(argo_data["LATs"]), np.isnan(argo_data["LONs"]))
    # interp LAT/ LON vals
    if len(np.where(location_mask == True)[0]) > 0:
        LATs, LONs = interp_missing_lat_lons(argo_data["LATs"], argo_data["LONs"], argo_data["JULDs"])
        argo_data["LATs"] = LATs
        argo_data["LONs"] = LONs
        argo_data["POSITION_QC"][location_mask] = 8
    
    return argo_data

# using JULD_LOCATION, we don't want to interp date, we want to fill w/ known val
def juld_check(argo_data):

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
    count_mask = np.logical_or(argo_data["NB_SAMPLE_CTD"] > 300,  np.logical_and(argo_data["NB_SAMPLE_CTD"] < 1, argo_data["NB_SAMPLE_CTD"] != -99))
    argo_data["PSAL_ADJUSTED_QC"][count_mask] = 3
    argo_data["CNDC_ADJUSTED_QC"][count_mask] = 3
    argo_data["TEMP_ADJUSTED_QC"][count_mask] = 3

    return argo_data

# NOTE: Use this for RBR inductive sensors to flag data too close to surface
def pres_depth_check(argo_data):
    pres_mask = np.where(argo_data["PRESs"] < 1)
    argo_data["PSAL_ADJUSTED_QC"][pres_mask] = 4
    argo_data["CNDC_ADJUSTED_QC"][pres_mask] = 4

    return argo_data

def process_autoset_qc_flags(qc_array, adjusted_qc_array, adjusted_data, label, profile_num, julds, pres_data):
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
    
    for i in np.arange(len(argo_data["PROFILE_NUMS"])):
        sal_checked = False
        temp_checked = False
        pres_checked = False
        trigger_ts = False

        # Check that QC_FLAG_CHECK has not been set yet
        if argo_data["QC_FLAG_CHECK"][i] == 0:

            # Condition to trigger data snapshot graph
            if ((not (np.all(argo_data["TEMP_QC"][i] == 0) or np.all(argo_data["TEMP_QC"][i] == 1))) and 
                (np.any(argo_data["TEMP_QC"][i] == 3) or np.any(argo_data["TEMP_QC"][i] == 4))) or \
                ((not (np.all(argo_data["PSAL_QC"][i] == 0) or np.all(argo_data["PSAL_QC"][i] == 1))) and 
                (np.any(argo_data["PSAL_QC"][i] == 3) or np.any(argo_data["PSAL_QC"][i] == 4))):
                data_snapshot_graph(argo_data, argo_data["PROFILE_NUMS"][i], first_time_run_module=True)

            pres_checked, argo_data["PRES_QC"][i], argo_data["PRES_ADJUSTED_QC"][i] = process_autoset_qc_flags(argo_data["PRES_QC"][i], argo_data["PRES_ADJUSTED_QC"][i], None, "PRES", argo_data["PROFILE_NUMS"][i], argo_data["JULDs"][i], argo_data["PRES_ADJUSTED"][i])
            temp_checked, argo_data["TEMP_QC"][i], argo_data["TEMP_ADJUSTED_QC"][i] = process_autoset_qc_flags(argo_data["TEMP_QC"][i], argo_data["TEMP_ADJUSTED_QC"][i], argo_data["TEMP_ADJUSTED"][i], "TEMP", argo_data["PROFILE_NUMS"][i], argo_data["JULDs"][i], argo_data["PRES_ADJUSTED"][i])
            sal_checked, argo_data["PSAL_QC"][i], argo_data["PSAL_ADJUSTED_QC"][i] = process_autoset_qc_flags(argo_data["PSAL_QC"][i], argo_data["PSAL_ADJUSTED_QC"][i], argo_data["PSAL_ADJUSTED"][i], "PSAL", argo_data["PROFILE_NUMS"][i], argo_data["JULDs"][i], argo_data["PRES_ADJUSTED"][i])
            # Trigger TS data graph if any flag required checking
            if temp_checked or sal_checked:
                trigger_ts = True
            
            if trigger_ts == True:
                selected_points = flag_TS_data_graphs(argo_data["PSAL_ADJUSTED"][i], argo_data["TEMP_ADJUSTED"][i], argo_data["JULDs"][i], argo_data["LONs"][i], argo_data["LATs"][i], argo_data["PRES_ADJUSTED"][i], argo_data["PROFILE_NUMS"][i], argo_data["TEMP_ADJUSTED_QC"][i], argo_data["PSAL_ADJUSTED_QC"][i])
                for j in np.arange(0, len(selected_points)):
                    index = selected_points[j]
                    # both points are bad
                    if index == 4:
                        argo_data["PSAL_ADJUSTED_QC"][i][j] = 4
                        argo_data["PSAL_QC"][i][j] = 4
                        argo_data["TEMP_ADJUSTED_QC"][i][j] = 4
                        argo_data["TEMP_QC"][i][j] = 4 
                    # sal is bad
                    elif index == 3:
                        argo_data["PSAL_ADJUSTED_QC"][i][j] = 4
                        argo_data["PSAL_QC"][i][j] = 4
                        argo_data["TEMP_ADJUSTED_QC"][i][j] = 1
                        argo_data["TEMP_QC"][i][j] = 1
                    # temp is bad
                    elif index == 2:
                        argo_data["PSAL_ADJUSTED_QC"][i][j] = 1
                        argo_data["PSAL_QC"][i][j] = 1
                        argo_data["TEMP_ADJUSTED_QC"][i][j] = 4
                        argo_data["TEMP_QC"][i][j] = 4
                    # index is 1, both points are good
                    else: 
                        argo_data["PSAL_ADJUSTED_QC"][i][j] = 1
                        argo_data["PSAL_QC"][i][j] = 1
                        argo_data["TEMP_ADJUSTED_QC"][i][j] = 1
                        argo_data["TEMP_QC"][i][j] = 1 
                print("Finished setting TEMP_QC and PSAL_QC")

            if sal_checked == True and temp_checked == True and pres_checked == True:
                argo_data["QC_FLAG_CHECK"][i] == 1

    return argo_data

def flag_data_points(argo_data, profile_num, data_type):

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

def flag_range_data(argo_data, profile_num, data_type):

    i = np.where(argo_data["PROFILE_NUMS"] == profile_num)[0][0]
    pres_arr = argo_data["PRES_ADJUSTED"][i]
    date = argo_data["JULDs"][i]

    if data_type == "PRES":
        qc_arr = argo_data["PRES_ADJUSTED_QC"][i]
        selected_points = flag_range_data_graphs(None, pres_arr, "PRES", qc_arr, profile_num, date)

    elif (data_type == "PSAL") or (data_type == "TEMP"):
        var_arr = argo_data[f"{data_type}_ADJUSTED"][i]
        qc_arr = argo_data[f"{data_type}_ADJUSTED_QC"][i]
        selected_points = flag_range_data_graphs(var_arr, pres_arr, data_type, qc_arr, profile_num, date)
    else:
        raise Exception("Invalid data_type")
    
    for index, (p1, p2) in enumerate(selected_points):
        if p2 is not None:
            for j in np.arange(p1, p2 + 1):
                argo_data[f"{data_type}_ADJUSTED_QC"][i][j] = 4
            print(f"Setting {data_type}_ADJUSTED_QC range {p1} - {p2} to BAD VAL")
        else:
            argo_data[f"{data_type}_ADJUSTED_QC"][i][p1] = 4
            print(f"Setting {data_type}_ADJUSTED_QC[{i}][{p1}] to BAD VAL")

    return argo_data

def flag_TS_data(argo_data, profile_num):

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

def data_snapshot_graph(argo_data, profile_num, first_time_run_module = None):

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
    axs[1, 1].text(0.5, 0.5, f'Datetime of Profile: {timestamp.date()} {timestamp.strftime('%H:%M:%S')}', fontsize=12, ha='center', va='center')
    axs[1, 1].text(0.5, 0.3, f'Lat: {lat:.2f} Lon: {lon:.2f}', fontsize=12, ha='center', va='center')
    if first_time_run_module:
        axs[1, 1].text(0.5, 0.1, f'Flag QC-point feature DISABLED for first time run', fontsize=12, ha='center', va='center')
    else:
        axs[1, 1].text(0.5, 0.1, f'Flag QC-point feature enabled for TEMP + PSAL graphs', fontsize=12, ha='center', va='center')
    axs[1, 1].axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    
    if not first_time_run_module:
        for j in np.arange(0, len(selected_points_temp)):
            index = selected_points_temp[j]
            argo_data[f"TEMP_ADJUSTED_QC"][i][j] = index
        for j in np.arange(0, len(selected_points_psal)):
            index = selected_points_psal[j]
            argo_data[f"PSAL_ADJUSTED_QC"][i][j] = index
        print("Finished setting TEMP_QC and PSAL_QC")
        
    return argo_data

def graph_pres_v_var_all(argo_data, data_type, use_adjusted, float_num, qc_arr_selection):

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

    selected_profiles = pres_v_var_all(pres, var, argo_data["JULDs"], argo_data["PROFILE_NUMS"], data_type, float_num)

    return selected_profiles

def graph_deep_section_var_all(argo_data, data_type, use_adjusted, float_num, qc_arr_selection):

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

    deep_section_var_all(pres, argo_data["JULDs"], var, float_num, data_type)

def graph_TS_all(argo_data, use_adjusted, float_num, qc_arr_selection):

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

    selected_profiles = TS_graph_single_dataset_all_profile(psal, temp, argo_data["JULDs"], argo_data["LONs"], argo_data["LATs"], pres, argo_data["PROFILE_NUMS"], float_num)
    
    return selected_profiles
   
def first_time_run(nc_filepath, dest_filepath, float_num):

    # Get dir of generated NETCDF files
    argo_data = read_intermediate_nc_file(nc_filepath)
    
    # CHECK 0: verify vals in [VAR]_QC arrs
    # NOTE:
    #  refer to argo_quality_control_manual: p.22
    #  be careful about flagging points as this method changes PAREM_QC arr as well as ADJUSTED_QC
    #  only flag points as bad if they are obviously/ near 100% bad??
    argo_data = verify_autoset_qc_flags(argo_data)
 
    # CHECK 1: Interpolate missing lat/lons and set QC flags
    # NOTE: passing in JULDs bc if lat/lon is missing -> JULD_LOCATION is missing
    argo_data = lat_lon_check(argo_data)

    # CHECK 2: fill-in times and set QC flag to 8
    argo_data = juld_check(argo_data)

    # CHECK 3: Set QC flags where counts are too high/low
    argo_data = count_check(argo_data)

    # Check 4: Set QC flags where PRES < 1m
    argo_data  = pres_depth_check(argo_data)

    # Write results back to NETCDF file
    make_intermediate_nc_file(argo_data, dest_filepath, float_num)  

def manipulate_data_flags(nc_filepath, dest_filepath, float_num, profile_num):
    
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

def generate_dataset_graphs(nc_filepath, dest_filepath, float_num, qc_arr_selection, data_type, use_adjusted):

    # Get dir of generated NETCDF files
    argo_data = read_intermediate_nc_file(nc_filepath)
    # get rid of all nan slices
    # argo_data = del_all_nan_slices(argo_data)

    selected_profiles_1 = graph_pres_v_var_all(argo_data, data_type, use_adjusted, float_num, qc_arr_selection)
    selected_profiles_2 = graph_TS_all(argo_data, use_adjusted, float_num, qc_arr_selection)
    graph_deep_section_var_all(argo_data, data_type, use_adjusted, float_num, qc_arr_selection)

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
            make_intermediate_nc_file(argo_data, dest_filepath, float_num, prof_num)  


def main():

    #nc_filepath = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\argo_to_nc\\F10051_1"
    nc_filepath = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\csv_to_nc\\F9186_1"
    float_num = "F9186"
    #dest_filepath = "c:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\argo_to_nc\\F10051_1"
    dest_filepath = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\csv_to_nc\\F9186_1"
    if not os.path.exists(dest_filepath):
        os.mkdir(dest_filepath)

    #first_time_run(nc_filepath, dest_filepath, float_num)

    profile_num = 88
    #manipulate_data_flags(nc_filepath, dest_filepath, float_num, profile_num)

    qc_arr_selection = [0, 1, 2] # only want good/ prob good data 
    data_type = "PSAL"                 # either PSAL or TEMP
    use_adjusted = True                
    generate_dataset_graphs(nc_filepath, dest_filepath, float_num, qc_arr_selection, data_type, use_adjusted)

if __name__ == '__main__':
 
    main()

    # MASTER PLAN 10/30/2024 - Sweet Zhang

    # READING DATA: CSV_TO_NC.PY MODULE
    # DONE: make var names more clear
    # DONE: init data arrs for various QC flags
    #    TEMP_ADJUSTED          PRES_ADJUSTED          PSAL_ADJUSTED          CNDC_ADJUSTED
    #    TEMP_ADJUSTED_ERROR    PRES_ADJUSTED_ERROR    PSAL_ADJUSTED_ERROR    CNDC_ADJUSTED_ERROR
    #    TEMP_ADJUSTED_QC       PRES_ADJUSTED_QC       PSAL_ADJUSTED_QC       CNDC_ADJUSTED_QC
        # NOTE: maybe add function in future to read PHY files... OR
        # maybe it's best to read in NETCDF files already on the website for the data?
        # but anyways, better to do all that here in the beginning - easier to change down the line
        
    # make changes to have these two portions read the files generated from CSV_TO_NC

    # DATA ANALYSIS PORTION: composed of graphing modules to allow person to look at the data
        # separate graphing module (graphs_nc.py) w/ variety of graphing tools
        # OVERALL TODOs:
        # --- clean up graphing modules
        # --- make it more user friendly...
        # --- incorp use of data flags IF PRESENT

    # DATA PROCESSING PORTION
        # so far... we have code in this module to:
        # Get raw CSV data and read in (LGR_CP_PTSCI) measurements: df_lats, df_lons, df_pres, df_temps, df_sals, df_juld, df_prof_nums, df_counts 
            # DONE: change this to read CSV_TO_NC generated output files
        # CLEAN UP DATA:
            # IMPLEMENT AUTO CHECKS:
                # DONE: Check CP_PTSCI COUNT number. throw away counts that are wayyyy to high
                    # are these responsible for the weird TS spikes @ the end of the profile?
                # DONE: Throw away all data that has a PRES of <1m
        # FIGURE OUT SALINITY DRIFT

    # WRITE FINAL PRODUCT TO NETCDF FILE
    """
    SCIENTIFIC_CALIB_EQUATION    # if no adjustments are made, var is filled by filled vals
                                 # if adjustments are made: PSAL_ADJUSTED = PSAL + CHANGE OF S etc
    SCIENTIFIC_CALIB_COEFFICIENT # if no adjustments are made, var is filled by filled vals
                                 # put what change of S is
    SCIENTIFIC_CALIB_COMMENT     # wording that describes the calibration, ex sensor drift detected etc
    SCIENTIFIC_CALIB_DATE # date of delayed-mode qc for EACH measurement parem YYYYMMDDHHMMSS

    DATA_MODE # changed to record 'D'
    DATA_STATE_INDICIATOR # record '2C' or '2C+'???
    DATE_UPDATE # record date of last update of netcdf file YYYYMMDDHHMMSS
    # name of single-profile ARGO netcdf file changed from R*.nc to D*.nc

    look in notebook for more data flags + info

    @ end read in value from {VAR_ADJUSTED_QC} -> copy into {VAR_QC}????
    """
