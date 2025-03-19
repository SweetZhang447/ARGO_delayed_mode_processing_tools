from datetime import datetime, timedelta, timezone
import numpy as np
import netCDF4 as nc4
import glob
import os
import itertools

def from_julian_day(julian_day):

    if not np.isnan(julian_day):

        reference_date = datetime(1950, 1, 1, 0, 0, 0, tzinfo=timezone.utc)     
        delta = timedelta(days=julian_day)
        dt = reference_date + delta

        return dt
    else:
        return julian_day

def to_julian_day(date_obj):

    if date_obj.tzinfo is None:
        date_obj = date_obj.replace(tzinfo=timezone.utc)

    delta = date_obj - datetime(1950, 1, 1, 0, 0, 0, tzinfo=timezone.utc) 
    julian_day = delta.total_seconds() / 86400

    return julian_day

def del_all_nan_slices(argo_data):
    
    pres_mask = np.isnan(argo_data["PRES_ADJUSTED"]).all(axis=1)
    temp_mask = np.isnan(argo_data["TEMP_ADJUSTED"]).all(axis=1)
    psal_mask = np.isnan(argo_data["PSAL_ADJUSTED"]).all(axis=1)
    # If ANY of these above data arrs are invalid, exclude the data
    bad_vals_mask =  ~(pres_mask | temp_mask | psal_mask)

    argo_data["PRES_ADJUSTED"] = argo_data["PRES_ADJUSTED"][bad_vals_mask]
    argo_data["TEMP_ADJUSTED"] = argo_data["TEMP_ADJUSTED"][bad_vals_mask]
    argo_data["PSAL_ADJUSTED"] = argo_data["PSAL_ADJUSTED"][bad_vals_mask]
    
    single_dim_bad_vals_mask = np.where(bad_vals_mask == True)
    argo_data["PROFILE_NUMS"] = argo_data["PROFILE_NUMS"][single_dim_bad_vals_mask]   
    argo_data["LATs"] = argo_data["LATs"][single_dim_bad_vals_mask]
    argo_data["LONs"] = argo_data["LONs"][single_dim_bad_vals_mask]
    argo_data["JULDs"] = argo_data["JULDs"][single_dim_bad_vals_mask]

    return argo_data

def make_intermediate_nc_file(argo_data, dest_filepath, float_num, profile_num = None):
    # if profile_num is not none, it means we want to generate a NC file for only 1 profile
    # in the dictionary of profiles 
    
    if profile_num is None:
        iterate_len_profile_nums = np.arange(len(argo_data["PROFILE_NUMS"]))
    else:
        iterate_len_profile_nums = [0]

    for i in iterate_len_profile_nums:
        
        if profile_num is None:
            prof_num = int(argo_data["PROFILE_NUMS"][i])
        else:
            i = np.where(argo_data["PROFILE_NUMS"] == profile_num)[0][0]
            prof_num = int(profile_num)

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

        cndc_var = nc.createVariable('CNDC', 'f4', 'records')
        cndc_var.units = "mhos/m"
        cndc_var[:] = argo_data["CNDCs"][i, :nan_index]

        temp_cndc_var = nc.createVariable('TEMP_CNDC', 'f4', 'records')
        temp_cndc_var.units = 'degree_celsius'
        temp_cndc_var[:] = argo_data["TEMP_CNDCs"][i, :nan_index]

        temp_cndc_qc_var = nc.createVariable('TEMP_CNDC_QC', 'f4', 'records')
        temp_cndc_qc_var[:] = argo_data["TEMP_CNDC_QC"][i, :nan_index]

        offset_var = nc.createVariable('PRES_OFFSET', 'f4', 'single_record')
        offset_var[:] = argo_data["PRES_OFFSET"][i]

        counts_var = nc.createVariable('NB_SAMPLE_CTD', 'f4', 'records')
        counts_var[:] = argo_data["NB_SAMPLE_CTD"][i, :nan_index]
        
        counts_qc_var = nc.createVariable('NB_SAMPLE_CTD_QC', 'f4', 'records')
        counts_qc_var[:] = argo_data["NB_SAMPLE_CTD_QC"][i, :nan_index]

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

        PSAL_ADJUSTED_QC_VAR = nc.createVariable('PSAL_ADJUSTED_QC', 'f4', 'records')
        PSAL_ADJUSTED_QC_VAR[:] = argo_data["PSAL_ADJUSTED_QC"][i, :nan_index]

        TEMP_ADJUSTED_VAR = nc.createVariable('TEMP_ADJUSTED', 'f4', 'records')
        TEMP_ADJUSTED_VAR[:] = argo_data["TEMP_ADJUSTED"][i, :nan_index]

        TEMP_ADJUSTED_QC_VAR = nc.createVariable('TEMP_ADJUSTED_QC', 'f4', 'records')
        TEMP_ADJUSTED_QC_VAR[:] = argo_data["TEMP_ADJUSTED_QC"][i, :nan_index]

        PRES_ADJUSTED_VAR = nc.createVariable('PRES_ADJUSTED', 'f4', 'records')
        PRES_ADJUSTED_VAR[:] = argo_data["PRES_ADJUSTED"][i, :nan_index]

        PRES_ADJUSTED_QC_VAR = nc.createVariable('PRES_ADJUSTED_QC', 'f4', 'records')
        PRES_ADJUSTED_QC_VAR[:] = argo_data["PRES_ADJUSTED_QC"][i, :nan_index]

        CNDC_ADJUSTED_VAR = nc.createVariable('CNDC_ADJUSTED', 'f4', 'records')
        CNDC_ADJUSTED_VAR[:] = argo_data["CNDC_ADJUSTED"][i, :nan_index]
        
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

        ptsci_timestamps_var = nc.createVariable('PTSCI_TIMESTAMPS', 'i8', 'records')
        ptsci_timestamps_var.long_name = "Format: YYYYMMDDHHMMSS"
        argo_data["PTSCI_TIMESTAMPS"][i, :nan_index][np.isnan(argo_data["PTSCI_TIMESTAMPS"][i, :nan_index])] = 0
        ptsci_timestamps_var[:] = argo_data["PTSCI_TIMESTAMPS"][i, :nan_index]

        nc.close()

def read_intermediate_nc_file(filepath):

    # Define the keys for argo data
    argo_keys = [
        "CNDC", "CNDC_ADJUSTED", "CNDC_ADJUSTED_QC", "CNDC_QC",
        "JULD", "JULD_LOCATION", "JULD_QC", "LAT", "LON", "NB_SAMPLE_CTD", "NB_SAMPLE_CTD_QC", "POSITION_QC",
        "PRES", "PRES_ADJUSTED", "PRES_ADJUSTED_QC", "PRES_OFFSET", "PRES_QC", 
        "PROFILE_NUMS", "QC_FLAG_CHECK",
        "PSAL", "PSAL_ADJUSTED", "PSAL_ADJUSTED_QC", "PSAL_QC",
        "TEMP", "TEMP_ADJUSTED", "TEMP_ADJUSTED_QC", "TEMP_CNDC", "TEMP_CNDC_QC", "TEMP_QC",
        "PTSCI_TIMESTAMPS"
    ]

    # Initialize argo_data dictionary with empty lists
    argo_data = {key: [] for key in argo_keys}

    # Get all NetCDF files in the specified filepath
    files = sorted(glob.glob(os.path.join(filepath, "*.nc")))

    for f in files:
        # Open NETCDF file and grab data
        float_dataset = nc4.Dataset(f)

        # Check if key variables have valid data
        PRES_temp = np.squeeze(float_dataset.variables['PRES'][:].filled(np.nan))
        PSAL_temp = np.squeeze(float_dataset.variables['PSAL'][:].filled(np.nan))
        TEMP_temp = np.squeeze(float_dataset.variables['TEMP'][:].filled(np.nan))

        if PRES_temp.size <= 2 or PSAL_temp.size <= 2 or TEMP_temp.size <= 2:
            print(f"Skipping file: {os.path.basename(f)} due to missing data")
        else:
            # Add data to the dictionary
            for key in argo_keys:
                if key in float_dataset.variables:
                    argo_data[key].append(float_dataset.variables[key][:].filled(np.nan))
                elif key == "PROFILE_NUMS":
                    argo_data[key].append(int(float_dataset.variables['PROFILE_NUM'][:].filled(np.nan)[0]))
        
        # Close the dataset after reading
        float_dataset.close()

    # Post-process: Combine arrays and handle missing values
    for key in argo_keys:
        if key in ["CNDC", "CNDC_ADJUSTED", "CNDC_ADJUSTED_QC", "CNDC_QC",
                   "NB_SAMPLE_CTD", "NB_SAMPLE_CTD_QC",
                   "PRES", "PRES_ADJUSTED", "PRES_ADJUSTED_QC", "PRES_QC",
                   "PSAL", "PSAL_ADJUSTED", "PSAL_ADJUSTED_QC", "PSAL_QC",
                   "TEMP", "TEMP_ADJUSTED", "TEMP_ADJUSTED_QC", "TEMP_CNDC", "TEMP_CNDC_QC", "TEMP_QC",
                   "PTSCI_TIMESTAMPS"
                   ]:
            argo_data[key] = np.squeeze(
                np.array(list(itertools.zip_longest(*argo_data[key], fillvalue=np.nan))).T
            )
        else:
            argo_data[key] = np.squeeze(np.array(argo_data[key]))

    # rename some keys
    rename_keys = ["CNDCs", "JULDs", "JULD_LOCATIONs", "LATs", "LONs",
                   "PRESs", "PSALs", "TEMPs", "TEMP_CNDCs"]
    for a in rename_keys:
        temp_keyname = a[:-1]
        argo_data[a] = argo_data.pop(temp_keyname)

    return argo_data
