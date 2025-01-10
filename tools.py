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

def read_intermediate_nc_file(filepath):

    # init temp arrs
    PROFILE_NUMS = []

    PRESs = []
    TEMPs = []
    PSALs = [] 
    COUNTs = []

    JULDs = [] 
    JULD_LOCATIONs = []
    LATs = []
    LONs = []
    JULD_QC = []
    POSITION_QC = []

    PSAL_ADJUSTED = []
    PSAL_ADJUSTED_ERROR = []
    PSAL_ADJUSTED_QC = []

    TEMP_ADJUSTED = []
    TEMP_ADJUSTED_ERROR = []
    TEMP_ADJUSTED_QC = []

    PRES_ADJUSTED = []
    PRES_ADJUSTED_ERROR = []
    PRES_ADJUSTED_QC = []

    CNDC_ADJUSTED_QC = []

    PSAL_QC = []
    TEMP_QC = []
    PRES_QC = []
    CNDC_QC = []

    QC_FLAG_CHECK = []
    
    files = sorted(glob.glob(os.path.join(filepath, "*.nc")))

    for f in files:

        # Open NETCDF file and grab data
        float_dataset = nc4.Dataset(f)

        PRES_temp = np.squeeze(float_dataset.variables['PRES'][:].filled(np.nan))
        PSAL_temp = np.squeeze(float_dataset.variables['PSAL'][:].filled(np.nan))
        TEMP_temp = np.squeeze(float_dataset.variables['TEMP'][:].filled(np.nan))

        if PRES_temp.size == 0 or PSAL_temp.size == 0 or TEMP_temp.size == 0 or PRES_temp.size == 1 or PSAL_temp.size == 1 or TEMP_temp.size == 1:
            print("Skipping file: {} due to missing data".format(os.path.basename(f)))
            # TODO: add in QC flags that properly mark data as bad
        else:
            # Read in Profile Numbers
            PROFILE_NUMS.append(int(float_dataset.variables['PROFILE_NUM'][:].filled(np.nan)[0]))
            
            # Read in Lat/Lons + QC flags 
            LATs.append(float_dataset.variables['LAT'][:].filled(np.nan))
            LONs.append(float_dataset.variables['LON'][:].filled(np.nan))
            JULDs.append(float_dataset.variables['JULD'][:].filled(np.nan))
            JULD_LOCATIONs.append(float_dataset.variables['JULD_LOCATION'][:].filled(np.nan))
            POSITION_QC.append(float_dataset.variables['POSITION_QC'][:].filled(np.nan))
            JULD_QC.append(float_dataset.variables['JULD_QC'][:].filled(np.nan))

            # Read in data arrs
            PRESs.append(float_dataset.variables['PRES'][:].filled(np.nan))
            TEMPs.append(float_dataset.variables['TEMP'][:].filled(np.nan))
            PSALs.append(float_dataset.variables['PSAL'][:].filled(np.nan))
            COUNTs.append(float_dataset.variables['COUNTS'][:].filled(np.nan))

            # Read in QC flags
            # NOTE: there shoudn't be any NANs since we set it in beginning to arr of 0s
            PSAL_ADJUSTED_QC.append(float_dataset.variables['PSAL_ADJUSTED_QC'][:].filled(np.nan))
            TEMP_ADJUSTED_QC.append(float_dataset.variables['TEMP_ADJUSTED_QC'][:].filled(np.nan))
            PRES_ADJUSTED_QC.append(float_dataset.variables['PRES_ADJUSTED_QC'][:].filled(np.nan))
            CNDC_ADJUSTED_QC.append(float_dataset.variables['CNDC_ADJUSTED_QC'][:].filled(np.nan))
            
            PSAL_ADJUSTED.append(float_dataset.variables['PSAL_ADJUSTED'][:].filled(np.nan))
            TEMP_ADJUSTED.append(float_dataset.variables['TEMP_ADJUSTED'][:].filled(np.nan))
            PRES_ADJUSTED.append(float_dataset.variables['PRES_ADJUSTED'][:].filled(np.nan))

            PSAL_ADJUSTED_ERROR.append(float_dataset.variables['PSAL_ADJUSTED_ERROR'][:].filled(np.nan))
            TEMP_ADJUSTED_ERROR.append(float_dataset.variables['TEMP_ADJUSTED_ERROR'][:].filled(np.nan))      
            PRES_ADJUSTED_ERROR.append(float_dataset.variables['PRES_ADJUSTED_ERROR'][:].filled(np.nan))
            
            PSAL_QC.append(float_dataset.variables['PSAL_QC'][:].filled(np.nan))
            TEMP_QC.append(float_dataset.variables['TEMP_QC'][:].filled(np.nan))
            PRES_QC.append(float_dataset.variables['PRES_QC'][:].filled(np.nan))
            CNDC_QC.append(float_dataset.variables['CNDC_QC'][:].filled(np.nan))

            QC_FLAG_CHECK.append(float_dataset.variables['QC_FLAG_CHECK'][:].filled(np.nan))
    

    PRESs = np.squeeze(np.array(list(itertools.zip_longest(*PRESs, fillvalue=np.nan))).T)
    TEMPs = np.squeeze(np.array(list(itertools.zip_longest(*TEMPs, fillvalue=np.nan))).T)
    PSALs = np.squeeze(np.array(list(itertools.zip_longest(*PSALs, fillvalue=np.nan))).T)
    COUNTs = np.squeeze(np.array(list(itertools.zip_longest(*COUNTs, fillvalue=np.nan))).T)

    JULDs = np.squeeze(np.array(JULDs))
    JULD_LOCATIONs = np.squeeze(np.array(JULD_LOCATIONs))
    LATs = np.squeeze(np.array(LATs))
    LONs = np.squeeze(np.array(LONs))
    JULD_QC = np.squeeze(np.array(JULD_QC))
    POSITION_QC = np.squeeze(np.array(POSITION_QC))

    PROFILE_NUMS = np.squeeze(np.array(PROFILE_NUMS))

    PSAL_ADJUSTED = np.squeeze(np.array(list(itertools.zip_longest(*PSAL_ADJUSTED, fillvalue=np.nan))).T)
    PSAL_ADJUSTED_ERROR = np.squeeze(np.array(list(itertools.zip_longest(*PSAL_ADJUSTED_ERROR, fillvalue=np.nan))).T)
    PSAL_ADJUSTED_QC = np.squeeze(np.array(list(itertools.zip_longest(*PSAL_ADJUSTED_QC, fillvalue=np.nan))).T)

    TEMP_ADJUSTED = np.squeeze(np.array(list(itertools.zip_longest(*TEMP_ADJUSTED, fillvalue=np.nan))).T)
    TEMP_ADJUSTED_ERROR = np.squeeze(np.array(list(itertools.zip_longest(*TEMP_ADJUSTED_ERROR, fillvalue=np.nan))).T)
    TEMP_ADJUSTED_QC = np.squeeze(np.array(list(itertools.zip_longest(*TEMP_ADJUSTED_QC, fillvalue=np.nan))).T)

    PRES_ADJUSTED = np.squeeze(np.array(list(itertools.zip_longest(*PRES_ADJUSTED, fillvalue=np.nan))).T)
    PRES_ADJUSTED_ERROR = np.squeeze(np.array(list(itertools.zip_longest(*PRES_ADJUSTED_ERROR, fillvalue=np.nan))).T)
    PRES_ADJUSTED_QC = np.squeeze(np.array(list(itertools.zip_longest(*PRES_ADJUSTED_QC, fillvalue=np.nan))).T)

    CNDC_ADJUSTED_QC = np.squeeze(np.array(list(itertools.zip_longest(*CNDC_ADJUSTED_QC, fillvalue=np.nan))).T)

    PSAL_QC = np.squeeze(np.array(list(itertools.zip_longest(*PSAL_QC, fillvalue=np.nan))).T)
    TEMP_QC = np.squeeze(np.array(list(itertools.zip_longest(*TEMP_QC, fillvalue=np.nan))).T)
    PRES_QC = np.squeeze(np.array(list(itertools.zip_longest(*PRES_QC, fillvalue=np.nan))).T)
    CNDC_QC = np.squeeze(np.array(list(itertools.zip_longest(*CNDC_QC, fillvalue=np.nan))).T)

    QC_FLAG_CHECK = np.squeeze(np.array(QC_FLAG_CHECK))

    argo_data = {
        "PRESs": PRESs,
        "TEMPs": TEMPs,
        "PSALs": PSALs,
        "COUNTs": COUNTs,
        "JULDs": JULDs,
        "JULD_LOCATIONs": JULD_LOCATIONs,
        "LATs": LATs,
        "LONs": LONs,
        "JULD_QC": JULD_QC,
        "POSITION_QC": POSITION_QC,
        "PSAL_ADJUSTED": PSAL_ADJUSTED,
        "PSAL_ADJUSTED_ERROR": PSAL_ADJUSTED_ERROR,
        "PSAL_ADJUSTED_QC": PSAL_ADJUSTED_QC,
        "TEMP_ADJUSTED": TEMP_ADJUSTED,
        "TEMP_ADJUSTED_ERROR": TEMP_ADJUSTED_ERROR,
        "TEMP_ADJUSTED_QC": TEMP_ADJUSTED_QC,
        "PRES_ADJUSTED": PRES_ADJUSTED,
        "PRES_ADJUSTED_ERROR": PRES_ADJUSTED_ERROR,
        "PRES_ADJUSTED_QC": PRES_ADJUSTED_QC,
        "PSAL_QC": PSAL_QC,
        "TEMP_QC": TEMP_QC,
        "PRES_QC": PRES_QC,
        "CNDC_QC": CNDC_QC,
        "PROFILE_NUMS": PROFILE_NUMS,
        "CNDC_ADJUSTED_QC": CNDC_ADJUSTED_QC,
        "QC_FLAG_CHECK": QC_FLAG_CHECK,
    }

    return argo_data

    """
    simplify code later...
    def read_nc_file(filepath):
    import itertools
    import numpy as np
    import glob
    import os
    import netCDF4 as nc4

    # Define the keys for argo data
    argo_keys = [
        "PRESs", "TEMPs", "PSALs", "COUNTs",
        "JULDs", "JULD_LOCATIONs", "LATs", "LONs",
        "JULD_QC", "POSITION_QC",
        "PSAL_ADJUSTED", "PSAL_ADJUSTED_ERROR", "PSAL_ADJUSTED_QC",
        "TEMP_ADJUSTED", "TEMP_ADJUSTED_ERROR", "TEMP_ADJUSTED_QC",
        "PRES_ADJUSTED", "PRES_ADJUSTED_ERROR", "PRES_ADJUSTED_QC",
        "PSAL_QC", "TEMP_QC", "PRES_QC", "CNDC_QC",
        "PROFILE_NUMS", "CNDC_ADJUSTED_QC", "QC_FLAG_CHECK",
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

        if any(arr.size <= 1 for arr in (PRES_temp, PSAL_temp, TEMP_temp)):
            print(f"Skipping file: {os.path.basename(f)} due to missing data")
            continue

        # Add data to the dictionary
        for key in argo_keys:
            if key in float_dataset.variables:
                argo_data[key].append(float_dataset.variables[key][:].filled(np.nan))
            elif key == "PROFILE_NUMS":
                argo_data[key].append(int(float_dataset.variables['PROFILE_NUM'][:].filled(np.nan)[0]))

    # Post-process: Combine arrays and handle missing values
    for key in argo_keys:
        if key in ["PRESs", "TEMPs", "PSALs", "COUNTs", "PSAL_ADJUSTED", "TEMP_ADJUSTED", "PRES_ADJUSTED",
                   "PSAL_ADJUSTED_ERROR", "TEMP_ADJUSTED_ERROR", "PRES_ADJUSTED_ERROR",
                   "PSAL_ADJUSTED_QC", "TEMP_ADJUSTED_QC", "PRES_ADJUSTED_QC",
                   "PSAL_QC", "TEMP_QC", "PRES_QC", "CNDC_QC", "CNDC_ADJUSTED_QC"]:
            argo_data[key] = np.squeeze(
                np.array(list(itertools.zip_longest(*argo_data[key], fillvalue=np.nan))).T
            )
        else:
            argo_data[key] = np.squeeze(np.array(argo_data[key]))

    return argo_data

    """