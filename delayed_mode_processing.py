import csv
import glob
import os 
import numpy as np
from datetime import datetime, timedelta
import itertools
import netCDF4 as nc4
import pandas as pd
import copy
from scipy.interpolate import interp1d
from graphs_nc import flag_TS_data_graphs, flag_range_data_graphs, verify_qc_flags_graphs
from tools import from_julian_day, to_julian_day, read_nc_file

def make_intermediate_nc_file(argo_data, dest_filepath, float_num):
    
    for i in np.arange(len(argo_data["PROFILE_NUMS"])):

        prof_num = int(argo_data["PROFILE_NUMS"][i])
        output_filename = os.path.join(dest_filepath, f"{float_num}-{prof_num:03}.nc")
        nc = nc4.Dataset(output_filename, 'w')

        # Set global attributes
        # TODO: make more detailed later
        nc.author = 'Sweet Zhang'

        # Get index to remove traling NaNs
        nan_index = np.where(~np.isnan(argo_data["PRESs"][i, :]))[0][-1] + 1

        # Create dimensions - name + length
        length = len(argo_data["PRESs"][i, :nan_index])
        record_dim = nc.createDimension('records', length)
        single_dim = nc.createDimension('single_record', 1)

        # create vars
        profile_nums_var = nc.createVariable('PROFILE_NUM', 'f4', 'single_record')
        profile_nums_var[:] = prof_num

        pressure_var = nc.createVariable('PRES', 'f4', 'records')
        pressure_var.units = 'DBAR'
        pressure_var[:] = argo_data["PRESs"][i, :nan_index]

        temperature_var = nc.createVariable('TEMP', 'f4', 'records')
        temperature_var.units = 'CELSIUS'
        temperature_var[:] = argo_data["TEMPs"][i, :nan_index]

        salinity_var = nc.createVariable('PSAL', 'f4', 'records')
        salinity_var.units = 'PSU'
        salinity_var[:] = argo_data["PSALs"][i, :nan_index]

        counts_var = nc.createVariable('COUNTS', 'f4', 'records')
        counts_var[:] = argo_data["COUNTs"][i, :nan_index]

        juld_var =  nc.createVariable('JULD', 'f4', 'single_record')
        juld_var[:] = argo_data["JULDs"][i]

        juld_location_var =  nc.createVariable('JULD_LOCATION', 'f4', 'single_record')
        juld_location_var[:] = argo_data["JULD_LOCATIONs"][i]

        lat_var = nc.createVariable('LAT', 'f4', 'single_record')
        lat_var[:] = argo_data["LATs"][i]

        lon_var = nc.createVariable('LON', 'f4', 'single_record')
        lon_var[:] = argo_data["LONs"][i]

        POSITION_QC_var = nc.createVariable('POSITION_QC', 'f4', 'single_record')
        POSITION_QC_var[:] = argo_data["POSITION_QC"][i]

        JULD_QC_var = nc.createVariable('JULD_QC', 'f4', 'single_record')
        JULD_QC_var[:] = argo_data["JULD_QC"][i]

        PSAL_ADJUSTED_VAR = nc.createVariable('PSAL_ADJUSTED', 'f4', 'records')
        PSAL_ADJUSTED_VAR[:] = argo_data["PSAL_ADJUSTED"][i, :nan_index]

        PSAL_ADJUSTED_ERROR_VAR = nc.createVariable('PSAL_ADJUSTED_ERROR', 'f4', 'records')
        PSAL_ADJUSTED_ERROR_VAR[:] = argo_data["PSAL_ADJUSTED_ERROR"][i, :nan_index]

        PSAL_ADJUSTED_QC_VAR = nc.createVariable('PSAL_ADJUSTED_QC', 'f4', 'records')
        PSAL_ADJUSTED_QC_VAR[:] = argo_data["PSAL_ADJUSTED_QC"][i, :nan_index]

        TEMP_ADJUSTED_VAR = nc.createVariable('TEMP_ADJUSTED', 'f4', 'records')
        TEMP_ADJUSTED_VAR[:] = argo_data["TEMP_ADJUSTED"][i, :nan_index]

        TEMP_ADJUSTED_ERROR_VAR = nc.createVariable('TEMP_ADJUSTED_ERROR', 'f4', 'records')
        TEMP_ADJUSTED_ERROR_VAR[:] = argo_data["TEMP_ADJUSTED_ERROR"][i, :nan_index]

        TEMP_ADJUSTED_QC_VAR = nc.createVariable('TEMP_ADJUSTED_QC', 'f4', 'records')
        TEMP_ADJUSTED_QC_VAR[:] = argo_data["TEMP_ADJUSTED_QC"][i, :nan_index]

        PRES_ADJUSTED_VAR = nc.createVariable('PRES_ADJUSTED', 'f4', 'records')
        PRES_ADJUSTED_VAR[:] = argo_data["PRES_ADJUSTED"][i, :nan_index]

        PRES_ADJUSTED_ERROR_VAR = nc.createVariable('PRES_ADJUSTED_ERROR', 'f4', 'records')
        PRES_ADJUSTED_ERROR_VAR[:] = argo_data["PRES_ADJUSTED_ERROR"][i, :nan_index]

        PRES_ADJUSTED_QC_VAR = nc.createVariable('PRES_ADJUSTED_QC', 'f4', 'records')
        PRES_ADJUSTED_QC_VAR[:] = argo_data["PRES_ADJUSTED_QC"][i, :nan_index]
        
        CNDC_ADJUSTED_QC_VAR = nc.createVariable('CNDC_ADJUSTED_QC', 'f4', 'records')
        CNDC_ADJUSTED_QC_VAR[:] = argo_data["CNDC_ADJUSTED_QC"][i, :nan_index]

        PSAL_QC_VAR = nc.createVariable('PSAL_QC', 'f4', 'records')
        PSAL_QC_VAR[:] = argo_data["PSAL_QC"][i, :nan_index]

        TEMP_QC_VAR = nc.createVariable('TEMP_QC', 'f4', 'records')
        TEMP_QC_VAR[:] = argo_data["TEMP_QC"][i, :nan_index]

        PRES_QC_VAR = nc.createVariable('PRES_QC', 'f4', 'records')
        PRES_QC_VAR[:] = argo_data["PRES_QC"][i, :nan_index]

        CNDC_QC_VAR = nc.createVariable('CNDC_QC', 'f4', 'records')
        CNDC_QC_VAR[:] = argo_data["CNDC_QC"][i, :nan_index]

        QC_FLAG_CHECK_VAR = nc.createVariable('QC_FLAG_CHECK', 'f4', 'single_record')
        QC_FLAG_CHECK_VAR[:] = argo_data["QC_FLAG_CHECK"][i]

        nc.close()

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
    if len(np.where(juld_mask == True)[0]) > 0:
        argo_data["JULDs"][juld_mask] = argo_data["JULD_LOCATIONs"][juld_mask]
        argo_data["JULD_QC"][juld_mask] = 8

    return argo_data

def count_check(argo_data):
    count_mask = np.logical_or(argo_data["COUNTs"] > 300,  np.logical_and(argo_data["COUNTs"] < 1, argo_data["COUNTs"] != -99))
    argo_data["PSAL_ADJUSTED_QC"][count_mask] = 4
    argo_data["CNDC_ADJUSTED_QC"][count_mask] = 4
    argo_data["TEMP_ADJUSTED_QC"][count_mask] = 4

    return argo_data

# NOTE: Use this for RBR inductive sensors to flag data too close to surface
def pres_depth_check(argo_data):
    pres_mask = np.where(argo_data["PRESs"] < 1)
    argo_data["PSAL_ADJUSTED_QC"][pres_mask] = 4
    argo_data["CNDC_ADJUSTED_QC"][pres_mask] = 4

    return argo_data

def set_adjusted_arrs(argo_data):

    # Step 2: get rid of QC flag = 4 vals
    argo_data["PSAL_ADJUSTED"][np.where(argo_data["PSAL_ADJUSTED_QC"] == 4)] = np.NaN
    argo_data["TEMP_ADJUSTED"][np.where(argo_data["TEMP_ADJUSTED_QC"] == 4)] = np.NaN
    argo_data["PRES_ADJUSTED"][np.where(argo_data["PRES_ADJUSTED_QC"] == 4)] = np.NaN

    return argo_data

def verify_qc_flags(argo_data):
    
    for i in np.arange(len(argo_data["PROFILE_NUMS"])):
        sal_checked = False
        temp_checked = False
        pres_checked = False
        trigger_ts = False
        # Check that QC_FLAG_CHECK has not been set yet
        if argo_data["QC_FLAG_CHECK"][i] == 0:
            # Check PRES arr first - check that QC arr is not all 0s or 1s
            if not (np.all(argo_data["PRES_QC"][i] == 0) or np.all(argo_data["PRES_QC"][i] == 1)):
                # Check that there are QC flags present to test
                if np.any(argo_data["PRES_QC"][i] == 3) or np.any(argo_data["PRES_QC"][i] == 4):
                    # Plot the ones that need to be checked
                    selected_indexes_qc, selected_indexes_arr_pts = verify_qc_flags_graphs(None, argo_data["PRES_ADJUSTED"][i], "PRES", argo_data["PRES_QC"][i], argo_data["PROFILE_NUMS"][i], argo_data["JULDs"][i])
                    # Get rid of marked indexes in QC arrs
                    for j in selected_indexes_qc:
                        argo_data["PRES_QC"][i][j] = 1   
                        argo_data["PRES_ADJUSTED_QC"][i][j] = 1
                        print(f"Setting PRES_QC[{i}][{j}] to GOOD VAL")  
                    # Mark bad points in arr
                    for j in selected_indexes_arr_pts:
                        argo_data["PRES_ADJUSTED_QC"][i][j] = 4
                        print(f"Setting PRES_ADJUSTED_QC[{i}][{j}] to BAD VAL")
                    pres_checked = True
                    trigger_ts = True
                else:
                    pres_checked = True
            else:
                pres_checked = True
          
            # Check temp
            if not (np.all(argo_data["TEMP_QC"][i] == 0) or np.all(argo_data["TEMP_QC"][i] == 1)):
                if np.any(argo_data["TEMP_QC"][i] == 3) or np.any(argo_data["TEMP_QC"][i] == 4):
                    selected_indexes_qc, selected_indexes_arr_pts = verify_qc_flags_graphs(argo_data["TEMP_ADJUSTED"][i], argo_data["PRES_ADJUSTED"][i], "TEMP", argo_data["TEMP_QC"][i], argo_data["PROFILE_NUMS"][i], argo_data["JULDs"][i])
                    for j in selected_indexes_qc:
                        argo_data["TEMP_QC"][i][j] = 1   
                        argo_data["TEMP_ADJUSTED_QC"][i][j] = 1  
                        print(f"Setting TEMP_QC[{i}][{j}] to GOOD VAL") 
                    for j in selected_indexes_arr_pts:
                        argo_data["TEMP_ADJUSTED_QC"][i][j] = 4
                        print(f"Setting TEMP_ADJUSTED_QC[{i}][{j}] to BAD VAL")
                    temp_checked = True
                    trigger_ts = True
                else:
                    temp_checked = True
            else:
                temp_checked = True
            
            # Check sal
            if not (np.all(argo_data["PSAL_QC"][i] == 0) or np.all(argo_data["PSAL_QC"][i] == 1)):
                if np.any(argo_data["PSAL_QC"][i] == 3) or np.any(argo_data["PSAL_QC"][i] == 4):
                    selected_indexes_qc, selected_indexes_arr_pts = verify_qc_flags_graphs(argo_data["PSAL_ADJUSTED"][i], argo_data["PRES_ADJUSTED"][i], "PSAL", argo_data["PSAL_QC"][i], argo_data["PROFILE_NUMS"][i], argo_data["JULDs"][i])
                    for j in selected_indexes_qc:
                        argo_data["PSAL_QC"][i][j] = 1  
                        argo_data["PSAL_ADJUSTED_QC"][i][j] = 1  
                        print(f"Setting PSAL_QC[{i}][{j}] to GOOD VAL") 
                    for j in selected_indexes_arr_pts:
                        argo_data["PSAL_ADJUSTED_QC"][i][j] = 4
                        print(f"Setting PSAL_ADJUSTED_QC[{i}][{j}] to BAD VAL")
                    sal_checked = True
                    trigger_ts = True
                else:
                    sal_checked = True
            else:
                sal_checked = True
            
            if trigger_ts == True:
        
                selected_points = flag_TS_data_graphs(argo_data["PSAL_ADJUSTED"][i], argo_data["TEMP_ADJUSTED"][i], argo_data["JULDs"][i], argo_data["LONs"][i], argo_data["LATs"][i], argo_data["PRES_ADJUSTED"][i], argo_data["PROFILE_NUMS"][i], argo_data["TEMP_QC"][i], argo_data["PSAL_QC"][i])
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
        selected_indexes_qc, selected_indexes_arr_pts = verify_qc_flags_graphs(None, pres_arr, "PRES", qc_arr, profile_num, date)
    
    elif (data_type == "PSAL") or (data_type == "TEMP"):
        var_arr = argo_data[f"{data_type}_ADJUSTED"][i]
        qc_arr = argo_data[f"{data_type}_ADJUSTED_QC"][i]
        selected_indexes_qc, selected_indexes_arr_pts = verify_qc_flags_graphs(var_arr, pres_arr, data_type, qc_arr, profile_num, date)
    else:
        raise Exception("Invalid data_type")
    
    for j in selected_indexes_qc:
        argo_data[f"{data_type}_ADJUSTED_QC"][i][j] = 1  
        print(f"Setting {data_type}_QC{i}[{j}] to GOOD VAL") 
    for j in selected_indexes_arr_pts:
        argo_data[f"{data_type}_ADJUSTED_QC"][i][j] = 4
        print(f"Setting {data_type}_ADJUSTED_QC{i}[{j}] to BAD VAL")

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
            print(f"Setting {data_type}_QC range {p1} - {p2} to BAD VAL")
        else:
            argo_data[f"{data_type}_ADJUSTED_QC"][i][p1] = 4
            print(f"Setting {data_type}_QC{i}[{p1}] to BAD VAL")

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

def manipulate_data():
    # Get dir of generated NETCDF files
    nc_filepath = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\argo_to_nc\\F10051_0"
    argo_data = read_nc_file(nc_filepath)
    profile_num = 86

    # Get rid of range of data
    argo_data = flag_range_data(argo_data, profile_num, "PRES")
    argo_data = flag_range_data(argo_data, profile_num, "PSAL")
    argo_data = flag_range_data(argo_data, profile_num, "TEMP")

    # TS diagram
    argo_data = flag_TS_data(argo_data, profile_num)

    # Flag individual data points
   
    argo_data = flag_data_points(argo_data, profile_num, "PRES")
    argo_data = flag_data_points(argo_data, profile_num, "PSAL")
    argo_data = flag_data_points(argo_data, profile_num, "TEMP")

    # Write results back to NETCDF file
    float_num = "F10015"
    dest_filepath = "c:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\argo_to_nc\\F10051_1"
    make_intermediate_nc_file(argo_data, dest_filepath, float_num)  


def first_time_run():

    # Get dir of generated NETCDF files
    nc_filepath = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\argo_to_nc\\F10051_0"
    argo_data = read_nc_file(nc_filepath)
    
    # CHECK 0: verify vals in [VAR]_QC arrs
    # NOTE:
    #   at this point the ADJUSTED arrs are just copies of the regular ones
    argo_data = verify_qc_flags(argo_data)
 
    # CHECK 1: Interpolate missing lat/lons and set QC flags
    # NOTE: passing in JULDs bc if lat/lon is missing -> JULD_LOCATION is missing
    argo_data = lat_lon_check(argo_data)

    # CHECK 2: fill-in times and set QC flag to 8
    argo_data = juld_check(argo_data)

    # CHECK 3: Set QC flags where counts are too high/low
    argo_data = count_check(argo_data)

    # Check 4: Set QC flags where PRES < 1m
    argo_data  = pres_depth_check(argo_data)

    # Use QC flags to set *VAR*_ADJUSTED arrays
    argo_data = set_adjusted_arrs(argo_data)

    # Write results back to NETCDF file
    float_num = "F10015"
    dest_filepath = "c:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\argo_to_nc\\F10051_1"
    make_intermediate_nc_file(argo_data, dest_filepath, float_num)  

def main():

    first_time_run()
    #manipulate_data()

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
