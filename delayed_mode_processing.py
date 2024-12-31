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

def make_nc_file(PRESs, TEMPs, PSALs, COUNTs, JULD_LOCATIONs, 
                 JULDs, LATs, LONs, JULD_QC, POSITION_QC, 
                 PSAL_ADJUSTED, PSAL_ADJUSTED_ERROR, PSAL_ADJUSTED_QC, 
                 TEMP_ADJUSTED, TEMP_ADJUSTED_ERROR, TEMP_ADJUSTED_QC, 
                 PRES_ADJUSTED, PRES_ADJUSTED_ERROR, PRES_ADJUSTED_QC,
                 PSAL_QC, TEMP_QC, PRES_QC, CNDC_QC, QC_FLAG_CHECK,
                 PROFILE_NUMS, CNDC_ADJUSTED_QC, dest_filepath, float_num):
    
    for i in np.arange(len(PROFILE_NUMS)):

        prof_num = int(PROFILE_NUMS[i])
        output_filename = os.path.join(dest_filepath, f"{float_num}-{prof_num:03}.nc")
        nc = nc4.Dataset(output_filename, 'w')

        # Set global attributes
        # TODO: make more detailed later
        nc.author = 'Sweet Zhang'

        # Get index to remove traling NaNs
        nan_index = np.where(~np.isnan(PRESs[i, :]))[0][-1] + 1

        # Create dimensions - name + length
        length = len(PRESs[i, :nan_index])
        record_dim = nc.createDimension('records', length)
        single_dim = nc.createDimension('single_record', 1)

        # create vars
        profile_nums_var = nc.createVariable('PROFILE_NUM', 'f4', 'single_record')
        profile_nums_var[:] = prof_num

        pressure_var = nc.createVariable('PRES', 'f4', 'records')
        pressure_var.units = 'DBAR'
        pressure_var[:] = PRESs[i, :nan_index]

        temperature_var = nc.createVariable('TEMP', 'f4', 'records')
        temperature_var.units = 'CELSIUS'
        temperature_var[:] = TEMPs[i, :nan_index]

        salinity_var = nc.createVariable('PSAL', 'f4', 'records')
        salinity_var.units = 'PSU'
        salinity_var[:] = PSALs[i, :nan_index]

        counts_var = nc.createVariable('COUNTS', 'f4', 'records')
        counts_var[:] = COUNTs[i, :nan_index]

        juld_var =  nc.createVariable('JULD', 'f4', 'single_record')
        juld_var[:] = JULDs[i]

        juld_location_var =  nc.createVariable('JULD_LOCATION', 'f4', 'single_record')
        juld_location_var[:] = JULD_LOCATIONs[i]

        lat_var = nc.createVariable('LAT', 'f4', 'single_record')
        lat_var[:] = LATs[i]

        lon_var = nc.createVariable('LON', 'f4', 'single_record')
        lon_var[:] = LONs[i]

        POSITION_QC_var = nc.createVariable('POSITION_QC', 'f4', 'single_record')
        POSITION_QC_var[:] = POSITION_QC[i]

        JULD_QC_var = nc.createVariable('JULD_QC', 'f4', 'single_record')
        JULD_QC_var[:] = JULD_QC[i]

        PSAL_ADJUSTED_VAR = nc.createVariable('PSAL_ADJUSTED', 'f4', 'records')
        PSAL_ADJUSTED_VAR[:] = PSAL_ADJUSTED[i, :nan_index]

        PSAL_ADJUSTED_ERROR_VAR = nc.createVariable('PSAL_ADJUSTED_ERROR', 'f4', 'records')
        PSAL_ADJUSTED_ERROR_VAR[:] = PSAL_ADJUSTED_ERROR[i, :nan_index]

        PSAL_ADJUSTED_QC_VAR = nc.createVariable('PSAL_ADJUSTED_QC', 'f4', 'records')
        PSAL_ADJUSTED_QC_VAR[:] = PSAL_ADJUSTED_QC[i, :nan_index]

        TEMP_ADJUSTED_VAR = nc.createVariable('TEMP_ADJUSTED', 'f4', 'records')
        TEMP_ADJUSTED_VAR[:] = TEMP_ADJUSTED[i, :nan_index]

        TEMP_ADJUSTED_ERROR_VAR = nc.createVariable('TEMP_ADJUSTED_ERROR', 'f4', 'records')
        TEMP_ADJUSTED_ERROR_VAR[:] = TEMP_ADJUSTED_ERROR[i, :nan_index]

        TEMP_ADJUSTED_QC_VAR = nc.createVariable('TEMP_ADJUSTED_QC', 'f4', 'records')
        TEMP_ADJUSTED_QC_VAR[:] = TEMP_ADJUSTED_QC[i, :nan_index]

        PRES_ADJUSTED_VAR = nc.createVariable('PRES_ADJUSTED', 'f4', 'records')
        PRES_ADJUSTED_VAR[:] = PRES_ADJUSTED[i, :nan_index]

        PRES_ADJUSTED_ERROR_VAR = nc.createVariable('PRES_ADJUSTED_ERROR', 'f4', 'records')
        PRES_ADJUSTED_ERROR_VAR[:] = PRES_ADJUSTED_ERROR[i, :nan_index]

        PRES_ADJUSTED_QC_VAR = nc.createVariable('PRES_ADJUSTED_QC', 'f4', 'records')
        PRES_ADJUSTED_QC_VAR[:] = PRES_ADJUSTED_QC[i, :nan_index]
        
        CNDC_ADJUSTED_QC_VAR = nc.createVariable('CNDC_ADJUSTED_QC', 'f4', 'records')
        CNDC_ADJUSTED_QC_VAR[:] = CNDC_ADJUSTED_QC[i, :nan_index]

        PSAL_QC_VAR = nc.createVariable('PSAL_QC', 'f4', 'records')
        PSAL_QC_VAR[:] = PSAL_QC[i, :nan_index]

        TEMP_QC_VAR = nc.createVariable('TEMP_QC', 'f4', 'records')
        TEMP_QC_VAR[:] = TEMP_QC[i, :nan_index]

        PRES_QC_VAR = nc.createVariable('PRES_QC', 'f4', 'records')
        PRES_QC_VAR[:] = PRES_QC[i, :nan_index]

        CNDC_QC_VAR = nc.createVariable('CNDC_QC', 'f4', 'records')
        CNDC_QC_VAR[:] = CNDC_QC[i, :nan_index]

        QC_FLAG_CHECK_VAR = nc.createVariable('QC_FLAG_CHECK', 'f4', 'single_record')
        QC_FLAG_CHECK_VAR[:] = QC_FLAG_CHECK[i]

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

def lat_lon_check(LATs, LONs, JULDs, POSITION_QC):
    
    location_mask = np.logical_or(np.isnan(LATs), np.isnan(LONs))
    # interp LAT/ LON vals
    if len(np.where(location_mask == True)[0]) > 0:
        LATs, LONs = interp_missing_lat_lons(LATs, LONs, JULDs)
        POSITION_QC[location_mask] = 8
    
    return LATs, LONs, POSITION_QC
    
def juld_check(JULDs, JULD_QC, JULD_LOCATIONs):

    juld_mask = np.isnan(JULDs)
    if len(np.where(juld_mask == True)[0]) > 0:
        JULDs[juld_mask] = JULD_LOCATIONs[juld_mask]
        JULD_QC[juld_mask] = 8

    return JULDs, JULD_QC

def count_check(COUNTs, PSAL_ADJUSTED_QC, TEMP_ADJUSTED_QC, CNDC_ADJUSTED_QC):
    count_mask = np.logical_or(COUNTs > 300,  np.logical_and(COUNTs < 1, COUNTs != -99))
    PSAL_ADJUSTED_QC[count_mask] = 4
    CNDC_ADJUSTED_QC[count_mask] = 4
    TEMP_ADJUSTED_QC[count_mask] = 4

    return PSAL_ADJUSTED_QC, TEMP_ADJUSTED_QC, CNDC_ADJUSTED_QC

# NOTE: Use this for RBR inductive sensors to flag data too close to surface
def pres_depth_check(PRESs, PSAL_ADJUSTED_QC, CNDC_ADJUSTED_QC):
    pres_mask = np.where(PRESs < 1)
    PSAL_ADJUSTED_QC[pres_mask] = 4
    CNDC_ADJUSTED_QC[pres_mask] = 4

    return PSAL_ADJUSTED_QC, CNDC_ADJUSTED_QC

def set_adjusted_arrs(PSAL_ADJUSTED, TEMP_ADJUSTED, PRES_ADJUSTED,
                      PSAL_ADJUSTED_QC, TEMP_ADJUSTED_QC, PRES_ADJUSTED_QC):

    # Step 2: get rid of QC flag = 4 vals
    PSAL_ADJUSTED[np.where(PSAL_ADJUSTED_QC == 4)] = np.NaN
    TEMP_ADJUSTED[np.where(TEMP_ADJUSTED_QC == 4)] = np.NaN
    PRES_ADJUSTED[np.where(PRES_ADJUSTED_QC == 4)] = np.NaN

    return PSAL_ADJUSTED, TEMP_ADJUSTED, PRES_ADJUSTED

def verify_qc_flags(PSAL_QC, TEMP_QC, PRES_QC, CNDC_QC, QC_FLAG_CHECK,
                    PSAL_ADJUSTED, TEMP_ADJUSTED, PRES_ADJUSTED, PROFILE_NUMS,
                    PSAL_ADJUSTED_QC, TEMP_ADJUSTED_QC, PRES_ADJUSTED_QC,
                    JULDs, LATs, LONs):
    
    for i in np.arange(len(PROFILE_NUMS)):
        sal_checked = False
        temp_checked = False
        pres_checked = False
        trigger_ts = False
        # Check that QC_FLAG_CHECK has not been set yet
        if QC_FLAG_CHECK[i] == 0:
            # Check PRES arr first - check that QC arr is not all 0s or 1s
            if not (np.all(PRES_QC[i] == 0) or np.all(PRES_QC[i] == 1)):
                # Check that there are QC flags present to test
                if np.any(PRES_QC[i] == 3) or np.any(PRES_QC[i] == 4):
                    # Plot the ones that need to be checked
                    selected_indexes_qc, selected_indexes_arr_pts = verify_qc_flags_graphs(None, PRES_ADJUSTED[i], "PRES", PRES_QC[i], PROFILE_NUMS[i], JULDs[i])
                    # Get rid of marked indexes in QC arrs
                    for j in selected_indexes_qc:
                        PRES_QC[i][j] = 1   
                        PRES_ADJUSTED_QC[i][j] = 1
                        print(f"Setting PRES_QC[{i}][{j}] to GOOD VAL")  
                    # Mark bad points in arr
                    for j in selected_indexes_arr_pts:
                        PRES_ADJUSTED_QC[i][j] = 4
                        print(f"Setting PRES_ADJUSTED_QC[{i}][{j}] to BAD VAL")
                    pres_checked = True
                    trigger_ts = True
                else:
                    pres_checked = True
            else:
                pres_checked = True
          
            # Check temp
            if not (np.all(TEMP_QC[i] == 0) or np.all(TEMP_QC[i] == 1)):
                if np.any(TEMP_QC[i] == 3) or np.any(TEMP_QC[i] == 4):
                    selected_indexes_qc, selected_indexes_arr_pts = verify_qc_flags_graphs(TEMP_ADJUSTED[i], PRES_ADJUSTED[i], "TEMP", TEMP_QC[i], PROFILE_NUMS[i], JULDs[i])
                    for j in selected_indexes_qc:
                        TEMP_QC[i][j] = 1   
                        TEMP_ADJUSTED_QC[i][j] = 1  
                        print(f"Setting TEMP_QC[{i}][{j}] to GOOD VAL") 
                    for j in selected_indexes_arr_pts:
                        TEMP_ADJUSTED_QC[i][j] = 4
                        print(f"Setting TEMP_ADJUSTED_QC[{i}][{j}] to BAD VAL")
                    temp_checked = True
                    trigger_ts = True
                else:
                    temp_checked = True
            else:
                temp_checked = True
            
            # Check sal
            if not (np.all(PSAL_QC[i] == 0) or np.all(PSAL_QC[i] == 1)):
                if np.any(PSAL_QC[i] == 3) or np.any(PSAL_QC[i] == 4):
                    selected_indexes_qc, selected_indexes_arr_pts = verify_qc_flags_graphs(PSAL_ADJUSTED[i], PRES_ADJUSTED[i], "PSAL", PSAL_QC[i], PROFILE_NUMS[i], JULDs[i])
                    for j in selected_indexes_qc:
                        PSAL_QC[i][j] = 1  
                        PSAL_ADJUSTED_QC[i][j] = 1  
                        print(f"Setting PSAL_QC[{i}][{j}] to GOOD VAL") 
                    for j in selected_indexes_arr_pts:
                        PSAL_ADJUSTED_QC[i][j] = 4
                        print(f"Setting PSAL_ADJUSTED_QC[{i}][{j}] to BAD VAL")
                    sal_checked = True
                    trigger_ts = True
                else:
                    sal_checked = True
            else:
                sal_checked = True
            
            if trigger_ts == True:
                selected_points = flag_TS_data_graphs(PSAL_ADJUSTED[i], TEMP_ADJUSTED[i], JULDs[i], LONs[i], LATs[i], PRES_ADJUSTED[i], PROFILE_NUMS[i], TEMP_QC[i], PSAL_QC[i])
                for j in np.arange(0, len(selected_points)):
                    index = selected_points[j]
                    # both points are bad
                    if index == 4:
                        PSAL_ADJUSTED_QC[i][j] = 4
                        PSAL_QC[i][j] = 4
                        TEMP_ADJUSTED_QC[i][j] = 4
                        TEMP_QC[i][j] = 4 
                    # sal is bad
                    elif index == 3:
                        PSAL_ADJUSTED_QC[i][j] = 4
                        PSAL_QC[i][j] = 4
                        TEMP_ADJUSTED_QC[i][j] = 1
                        TEMP_QC[i][j] = 1
                    # temp is bad
                    elif index == 2:
                        PSAL_ADJUSTED_QC[i][j] = 1
                        PSAL_QC[i][j] = 1
                        TEMP_ADJUSTED_QC[i][j] = 4
                        TEMP_QC[i][j] = 4
                    # index is 1, both points are good
                    else: 
                        PSAL_ADJUSTED_QC[i][j] = 1
                        PSAL_QC[i][j] = 1
                        TEMP_ADJUSTED_QC[i][j] = 1
                        TEMP_QC[i][j] = 1 
                print("Finished setting TEMP_QC and PSAL_QC")

            if sal_checked == True and temp_checked == True and pres_checked == True:
                QC_FLAG_CHECK[i] == 1

    return QC_FLAG_CHECK, PSAL_QC, TEMP_QC, PRES_QC, PSAL_ADJUSTED_QC, TEMP_ADJUSTED_QC, PRES_ADJUSTED_QC

def flag_data_points(var_arr, pres_arr, data_type, qc_var_arr, JULDs, PROFILE_NUMS, profile_num):

    i =  np.where(PROFILE_NUMS == profile_num)[0][0]
    pres_arr_single_prof = np.squeeze(pres_arr[i])
    qc_var_arr_single_prof =  np.squeeze(qc_var_arr[i])
    date = np.squeeze(JULDs[i])

    if data_type == "PRES":
        selected_indexes_qc, selected_indexes_arr_pts = verify_qc_flags_graphs(None, pres_arr_single_prof, "PRES", qc_var_arr_single_prof, profile_num, date)
    else:
        var_arr_single_prof = np.squeeze(var_arr[i])
        selected_indexes_qc, selected_indexes_arr_pts = verify_qc_flags_graphs(var_arr_single_prof, pres_arr_single_prof, data_type, qc_var_arr_single_prof, profile_num, date)
    
    for j in selected_indexes_qc:
        qc_var_arr[i][j] = 1  
        print(f"Setting {data_type}_QC{i}[{j}] to GOOD VAL") 
    for j in selected_indexes_arr_pts:
        qc_var_arr[i][j] = 4
        print(f"Setting {data_type}_ADJUSTED_QC{i}[{j}] to BAD VAL")

    return qc_var_arr

def flag_range_data(var_arr, pres_arr, data_type, qc_var_arr, JULDs, PROFILE_NUMS, profile_num):

    i = np.where(PROFILE_NUMS == profile_num)[0][0]
    pres_arr_single_prof = np.squeeze(pres_arr[i])
    qc_var_arr_single_prof =  np.squeeze(qc_var_arr[i])
    date = np.squeeze(JULDs[i])

    if data_type == "PRES":
        selected_points = flag_range_data_graphs(None, pres_arr_single_prof, "PRES", qc_var_arr_single_prof, profile_num, date)
    else:
        var_arr_single_prof = np.squeeze(var_arr[i])
        selected_points = flag_range_data_graphs(var_arr_single_prof, pres_arr_single_prof, data_type, qc_var_arr_single_prof, profile_num, date)
    
    for index, (p1, p2) in enumerate(selected_points):
        if p2 is not None:
            for j in np.arange(p1, p2 + 1):
                qc_var_arr[i][j] = 4
            print(f"Setting {data_type}_QC range {p1} - {p2} to BAD VAL")
        else:
            qc_var_arr[i][p1] = 4
            print(f"Setting {data_type}_QC{i}[{p1}] to BAD VAL")

    return qc_var_arr

def flag_TS_data(df_SALs, df_TEMPs, df_JULD, df_LONs, df_LATs, df_PRESs, PROFILE_NUMS, profile_num,
                 TEMP_ADJUSTED_QC, PSAL_ADJUSTED_QC):

    i = np.where(PROFILE_NUMS == profile_num)[0][0]
    sal_arr = np.squeeze(df_SALs[i])
    temp_arr = np.squeeze(df_TEMPs[i])
    pres_arr = np.squeeze(df_PRESs[i])
    temp_qc_single_prof = np.squeeze(TEMP_ADJUSTED_QC[i]) 
    psal_qc_single_prof = np.squeeze(PSAL_ADJUSTED_QC[i])
    lons = df_LONs[i]
    lats = df_LATs[i]
    juld = np.squeeze(df_JULD[i])

    selected_points = flag_TS_data_graphs(sal_arr, temp_arr, juld, lons, lats, pres_arr, profile_num, temp_qc_single_prof, psal_qc_single_prof)

    for j in np.arange(0, len(selected_points)):
        index = selected_points[j]
        # both points are bad
        if index == 4:
            PSAL_ADJUSTED_QC[i][j] = 4
            TEMP_ADJUSTED_QC[i][j] = 4
        # sal is bad
        elif index == 3:
            PSAL_ADJUSTED_QC[i][j] = 4
            TEMP_ADJUSTED_QC[i][j] = 1
        # temp is bad
        elif index == 2:
            PSAL_ADJUSTED_QC[i][j] = 1
            TEMP_ADJUSTED_QC[i][j] = 4
        # index is 1, both points are good
        else: 
            PSAL_ADJUSTED_QC[i][j] = 1
            TEMP_ADJUSTED_QC[i][j] = 1
    
    print("Finished setting TEMP_QC and PSAL_QC")
       
    return PSAL_ADJUSTED_QC, TEMP_ADJUSTED_QC

def manipulate_data():
    # Get dir of generated NETCDF files
    nc_filepath = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\argo_to_nc\\F10051_0"

    (PRESs, TEMPs, PSALs, COUNTs, 
     JULDs, JULD_LOCATIONs, LATs, LONs, JULD_QC, POSITION_QC, 
     PSAL_ADJUSTED, PSAL_ADJUSTED_ERROR, PSAL_ADJUSTED_QC, 
     TEMP_ADJUSTED, TEMP_ADJUSTED_ERROR, TEMP_ADJUSTED_QC, 
     PRES_ADJUSTED, PRES_ADJUSTED_ERROR, PRES_ADJUSTED_QC,
     PSAL_QC, TEMP_QC, PRES_QC, CNDC_QC,
     PROFILE_NUMS, CNDC_ADJUSTED_QC, QC_FLAG_CHECK) = read_nc_file(nc_filepath)
    
    # Get rid of range of data
    PRES_ADJUSTED_QC = flag_range_data(None, PRES_ADJUSTED, "PRES", PRES_ADJUSTED_QC, JULDs, PROFILE_NUMS, 86)
    # PSAL_ADJUSTED_QC = flag_range_data(PSAL_ADJUSTED, PRES_ADJUSTED, "PSAL", PSAL_ADJUSTED_QC, JULDs, PROFILE_NUMS, 86)

    PSAL_ADJUSTED_QC, TEMP_ADJUSTED_QC = flag_TS_data(PSAL_ADJUSTED, TEMP_ADJUSTED, JULDs, LONs, LATs, PRES_ADJUSTED, PROFILE_NUMS, 86,
                                                      TEMP_ADJUSTED_QC, PSAL_ADJUSTED_QC)

    # Flag individual data points
    PSAL_ADJUSTED_QC = flag_data_points(PSAL_ADJUSTED, PRES_ADJUSTED, "PSAL", PSAL_ADJUSTED_QC, JULDs, PROFILE_NUMS, 86)
    TEMP_ADJUSTED_QC = flag_data_points(TEMP_ADJUSTED, PRES_ADJUSTED, "TEMP", TEMP_ADJUSTED_QC, JULDs, PROFILE_NUMS, 86)
    # PRES_ADJUSTED_QC = flag_data_points(None, PRES_ADJUSTED, "PRES", PRES_ADJUSTED_QC, JULDs, PROFILE_NUMS, 86)
    
    raise Exception
    # Write results back to NETCDF file
    float_num = "F10015"
    dest_filepath = "c:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\argo_to_nc\\F10051_1"
    make_nc_file(PRESs, TEMPs, PSALs, COUNTs, JULD_LOCATIONs,
                 JULDs, LATs, LONs, JULD_QC, POSITION_QC, 
                 PSAL_ADJUSTED, PSAL_ADJUSTED_ERROR, PSAL_ADJUSTED_QC, 
                 TEMP_ADJUSTED, TEMP_ADJUSTED_ERROR, TEMP_ADJUSTED_QC, 
                 PRES_ADJUSTED, PRES_ADJUSTED_ERROR, PRES_ADJUSTED_QC,
                 PSAL_QC, TEMP_QC, PRES_QC, CNDC_QC, QC_FLAG_CHECK,
                 PROFILE_NUMS, CNDC_ADJUSTED_QC, dest_filepath, float_num)


def first_time_run():

    # Get dir of generated NETCDF files
    nc_filepath = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\argo_to_nc\\F10051_0"

    (PRESs, TEMPs, PSALs, COUNTs, 
     JULDs, JULD_LOCATIONs, LATs, LONs, JULD_QC, POSITION_QC, 
     PSAL_ADJUSTED, PSAL_ADJUSTED_ERROR, PSAL_ADJUSTED_QC, 
     TEMP_ADJUSTED, TEMP_ADJUSTED_ERROR, TEMP_ADJUSTED_QC, 
     PRES_ADJUSTED, PRES_ADJUSTED_ERROR, PRES_ADJUSTED_QC,
     PSAL_QC, TEMP_QC, PRES_QC, CNDC_QC,
     PROFILE_NUMS, CNDC_ADJUSTED_QC, QC_FLAG_CHECK) = read_nc_file(nc_filepath)
    
    # CHECK 0: verify vals in [VAR]_QC arrs
    # NOTE:
    #   at this point the ADJUSTED arrs are just copies of the regular ones
    (QC_FLAG_CHECK, PSAL_QC, TEMP_QC, PRES_QC,
     PSAL_ADJUSTED_QC, TEMP_ADJUSTED_QC, PRES_ADJUSTED_QC) = verify_qc_flags(PSAL_QC, TEMP_QC, PRES_QC, CNDC_QC, QC_FLAG_CHECK,
                                                               PSAL_ADJUSTED, TEMP_ADJUSTED, PRES_ADJUSTED, PROFILE_NUMS,
                                                               PSAL_ADJUSTED_QC, TEMP_ADJUSTED_QC, PRES_ADJUSTED_QC,
                                                               JULDs, LATs, LONs)

    # CHECK 1: Interpolate missing lat/lons and set QC flags
    # NOTE: passing in JULDs bc if lat/lon is missing -> JULD_LOCATION is missing
    LATs, LONs, POSITION_QC = lat_lon_check(LATs, LONs, JULDs, POSITION_QC)

    # CHECK 2: Interpolate times and set QC flag to 8
    # NOTE: passing in JULD_LOCATION, we don't want to interp date, we want to fill w/ known val
    JULDs, JULD_QC = juld_check(JULDs, JULD_QC, JULD_LOCATIONs)

    # CHECK 3: Set QC flags where counts are too high/low
    PSAL_ADJUSTED_QC, TEMP_ADJUSTED_QC, CNDC_ADJUSTED_QC = count_check(COUNTs, PSAL_ADJUSTED_QC, TEMP_ADJUSTED_QC, CNDC_ADJUSTED_QC)

    # Check 4: Set QC flags where PRES < 1m
    PSAL_ADJUSTED_QC, CNDC_ADJUSTED_QC  = pres_depth_check(PRESs, PSAL_ADJUSTED_QC, CNDC_ADJUSTED_QC)

    # Step 2:
    # clean up data points + passing in said QC flags above 
    # flag_bad_data(df_lats, df_lons, df_pres, df_temps, df_sals, df_juld, df_prof_nums, df_counts)

    # Use QC flags to set *VAR*_ADJUSTED arrays
    PSAL_ADJUSTED, TEMP_ADJUSTED, PRES_ADJUSTED = set_adjusted_arrs(PSAL_ADJUSTED, TEMP_ADJUSTED, PRES_ADJUSTED,
                                                                    PSAL_ADJUSTED_QC, TEMP_ADJUSTED_QC, PRES_ADJUSTED_QC)

    # Write results back to NETCDF file
    float_num = "F10015"
    dest_filepath = "c:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\argo_to_nc\\F10051_1"
    make_nc_file(PRESs, TEMPs, PSALs, COUNTs, JULD_LOCATIONs,
                 JULDs, LATs, LONs, JULD_QC, POSITION_QC, 
                 PSAL_ADJUSTED, PSAL_ADJUSTED_ERROR, PSAL_ADJUSTED_QC, 
                 TEMP_ADJUSTED, TEMP_ADJUSTED_ERROR, TEMP_ADJUSTED_QC, 
                 PRES_ADJUSTED, PRES_ADJUSTED_ERROR, PRES_ADJUSTED_QC,
                 PSAL_QC, TEMP_QC, PRES_QC, CNDC_QC, QC_FLAG_CHECK,
                 PROFILE_NUMS, CNDC_ADJUSTED_QC, dest_filepath, float_num)  

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
