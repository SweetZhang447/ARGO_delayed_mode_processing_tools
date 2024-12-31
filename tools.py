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


def read_nc_file(filepath):

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

    return (PRESs, TEMPs, PSALs, COUNTs, 
            JULDs, JULD_LOCATIONs, LATs, LONs, JULD_QC, POSITION_QC, 
            PSAL_ADJUSTED, PSAL_ADJUSTED_ERROR, PSAL_ADJUSTED_QC, 
            TEMP_ADJUSTED, TEMP_ADJUSTED_ERROR, TEMP_ADJUSTED_QC, 
            PRES_ADJUSTED, PRES_ADJUSTED_ERROR, PRES_ADJUSTED_QC,
            PSAL_QC, TEMP_QC, PRES_QC, CNDC_QC,
            PROFILE_NUMS, CNDC_ADJUSTED_QC, QC_FLAG_CHECK)

