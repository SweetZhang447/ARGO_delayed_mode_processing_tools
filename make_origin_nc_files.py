"""
make_origin_nc_files.py — Ingestion layer: convert raw float data or real-time ARGO netCDF to intermediate format.

This is the first step in the DMODE pipeline. It reads either:
  a) Raw float science_log.csv + system_log.txt files from RBR-equipped floats, or
  b) Real-time ARGO netCDF profile files downloaded from the ARGO data center.

All output is written in the intermediate netCDF format defined in tools.py, with one
.nc file per profile, named {float_num}-{profile_num:03}.nc.

Usage: configure the flags in main() and run this script directly.

Control flags in main()
-----------------------
download_ARGO_NETCDF_files : int (0 or 1)
    1 = download real-time ARGO netCDF files from dload_url before processing.
read_ARGO_NETCDF_files : int (0 or 1)
    1 = read and convert real-time ARGO netCDF files from input_dir.
read_RAW_CSV_files : int (0 or 1)
    1 = read and convert raw RBR float CSV/TXT log files from input_dir.
"""
import csv
import glob
import os 
import netCDF4 as nc4
import numpy as np
from datetime import datetime, timezone, timedelta
from tools import from_julian_day, to_julian_day
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor
import requests
from pathlib import Path
import copy

def make_nc_file_origin(profile_num, pressures, temps, sals, cndc, temp_cndc, counts,
                        PRES_ADJUSTED, TEMP_ADJUSTED, PSAL_ADJUSTED,
                        latitude, longitude, juld_timestamp, juld_location, dest_filepath, float_num,
                        **kwargs):
    """
    Write one intermediate netCDF file for a single profile.

    Creates a file at dest_filepath/{float_num}-{profile_num:03}.nc with the
    standard intermediate netCDF dimensions (records, single_record) and all
    required variables. Any QC arrays not provided via kwargs are initialized to
    arrays of zeros (no QC applied).

    Parameters
    ----------
    profile_num : int
        Profile number, used in the filename and stored as PROFILE_NUM.
    pressures, temps, sals, cndc, temp_cndc : array-like
        Raw sensor data arrays (depth levels).
    counts : array-like
        NB_SAMPLE_CTD bin-average sample counts per depth level.
    PRES_ADJUSTED, TEMP_ADJUSTED, PSAL_ADJUSTED : array-like
        Adjusted data arrays (same shape as raw; PRES_ADJUSTED = PRES - surface offset).
    latitude, longitude : float
        Profile position.
    juld_timestamp : float
        Julian day of the profile (1950-01-01 reference).
    juld_location : float
        Julian day of the GPS position fix.
    dest_filepath : str
        Output directory.
    float_num : str
        Float identifier (e.g. 'F9186').
    **kwargs : optional
        QC and metadata arrays to include. If not provided, initialized to zeros:
        PRES_QC, TEMP_QC, PSAL_QC, CNDC_QC, TEMP_CNDC_QC, NB_SAMPLE_CTD_QC,
        PSAL_ADJUSTED_QC, TEMP_ADJUSTED_QC, PRES_ADJUSTED_QC, POSITION_QC,
        JULD_QC, PRES_OFFSET, PTSCI_TIMESTAMPS.
    """

    output_filename = os.path.join(dest_filepath, f"{float_num}-{profile_num:03}.nc")
    nc = nc4.Dataset(output_filename, 'w')

    # Set global attributes
    nc.author = 'Sweet Zhang'
    nc.summary = 'NC file: LGR_CP_PTSCI readings ONLY'

    # convert all lists to numpy array types
    pressures = np.asarray(pressures)
    temps = np.asarray(temps)
    sals = np.asarray(sals)
    cndc = np.asarray(cndc)
    temp_cndc = np.asarray(temp_cndc)
    counts = np.asarray(counts)
    
    # Initialize variables with values passed through kwargs, or use defaults if not provided
    PSAL_ADJUSTED_QC = kwargs.get("PSAL_ADJUSTED_QC", np.full(sals.shape, fill_value=0))
    TEMP_ADJUSTED_QC = kwargs.get("TEMP_ADJUSTED_QC", np.full(temps.shape, fill_value=0))
    PRES_ADJUSTED_QC = kwargs.get("PRES_ADJUSTED_QC", np.full(pressures.shape, fill_value=0))
 
    PSAL_QC = kwargs.get("PSAL_QC", np.full(pressures.shape, fill_value=0)) 
    TEMP_QC = kwargs.get("TEMP_QC", np.full(pressures.shape, fill_value=0)) 
    PRES_QC = kwargs.get("PRES_QC", np.full(pressures.shape, fill_value=0)) 
    CNDC_QC = kwargs.get("CNDC_QC", np.full(pressures.shape, fill_value=0)) 

    TEMP_CNDC_QC = kwargs.get("TEMP_CNDC_QC", np.full(pressures.shape, fill_value=0)) 
    NB_SAMPLE_CTD_QC = kwargs.get("NB_SAMPLE_CTD_QC", np.full(pressures.shape, fill_value=0)) 
    POSITION_QC = kwargs.get("POSITION_QC", np.nan)
    JULD_QC = kwargs.get("JULD_QC", np.nan)
    offset = kwargs.get("pres_offset", None)
    PTSCI_TIMESTAMPS = kwargs.get("PTSCI_TIMESTAMPS", np.full(pressures.shape, fill_value=0)) 

    # Create dimensions - name + length
    length = pressures.size
    record_dim = nc.createDimension('records', length)
    lat_dim = nc.createDimension('single_record', 1)

    # create vars
    profile_nums_var = nc.createVariable('PROFILE_NUM', 'f4', 'single_record')
    profile_nums_var[:] = profile_num

    pressure_var = nc.createVariable('PRES', 'f4', 'records')
    pressure_var.units = 'DBAR'
    pressure_var[:] = pressures

    temperature_var = nc.createVariable('TEMP', 'f4', 'records')
    temperature_var.units = 'CELSIUS'
    temperature_var[:] = temps

    salinity_var = nc.createVariable('PSAL', 'f4', 'records')
    salinity_var.units = 'PSU'
    salinity_var[:] = sals

    cndc_var = nc.createVariable('CNDC', 'f4', 'records')
    cndc_var.units = "mhos/m"
    cndc_var[:] = cndc

    temp_cndc_var = nc.createVariable('TEMP_CNDC', 'f4', 'records')
    temp_cndc_var.units = 'degree_celsius'
    temp_cndc_var[:] = temp_cndc

    temp_cndc_qc_var = nc.createVariable('TEMP_CNDC_QC', 'f4', 'records')
    temp_cndc_qc_var[:] = TEMP_CNDC_QC

    offset_var = nc.createVariable('PRES_OFFSET', 'f4', 'single_record')
    if offset == None:
        offset_var[:] = int(-9999)
    else:
        offset_var[:] = offset
    
    ptsci_timestamps_var = nc.createVariable('PTSCI_TIMESTAMPS', 'i8', 'records')
    ptsci_timestamps_var.long_name = "Format: YYYYMMDDHHMMSS"
    ptsci_timestamps_var[:] = PTSCI_TIMESTAMPS

    counts_var = nc.createVariable('NB_SAMPLE_CTD', 'f4', 'records')
    counts_var[:] = counts

    counts_qc_var = nc.createVariable('NB_SAMPLE_CTD_QC', 'f4', 'records')
    counts_qc_var[:] = NB_SAMPLE_CTD_QC

    juld_var =  nc.createVariable('JULD', 'f4', 'single_record')
    juld_var[:] = juld_timestamp

    juld_location_var =  nc.createVariable('JULD_LOCATION', 'f4', 'single_record')
    juld_location_var[:] = juld_location

    lat_var = nc.createVariable('LAT', 'f4', 'single_record')
    lat_var[:] = latitude

    lon_var = nc.createVariable('LON', 'f4', 'single_record')
    lon_var[:] = longitude

    POSITION_QC_var = nc.createVariable('POSITION_QC', 'f4', 'single_record')
    POSITION_QC_var[:] = POSITION_QC

    JULD_QC_var = nc.createVariable('JULD_QC', 'f4', 'single_record')
    JULD_QC_var[:] = JULD_QC

    PSAL_ADJUSTED_VAR = nc.createVariable('PSAL_ADJUSTED', 'f4', 'records')
    PSAL_ADJUSTED_VAR[:] = PSAL_ADJUSTED

    PSAL_ADJUSTED_QC_VAR = nc.createVariable('PSAL_ADJUSTED_QC', 'f4', 'records')
    PSAL_ADJUSTED_QC_VAR[:] = PSAL_ADJUSTED_QC

    TEMP_ADJUSTED_VAR = nc.createVariable('TEMP_ADJUSTED', 'f4', 'records')
    TEMP_ADJUSTED_VAR[:] = TEMP_ADJUSTED

    TEMP_ADJUSTED_QC_VAR = nc.createVariable('TEMP_ADJUSTED_QC', 'f4', 'records')
    TEMP_ADJUSTED_QC_VAR[:] = TEMP_ADJUSTED_QC

    PRES_ADJUSTED_VAR = nc.createVariable('PRES_ADJUSTED', 'f4', 'records')
    PRES_ADJUSTED_VAR[:] = PRES_ADJUSTED

    PRES_ADJUSTED_QC_VAR = nc.createVariable('PRES_ADJUSTED_QC', 'f4', 'records')
    PRES_ADJUSTED_QC_VAR[:] = PRES_ADJUSTED_QC

    PSAL_QC_VAR = nc.createVariable('PSAL_QC', 'f4', 'records')
    PSAL_QC_VAR[:] = PSAL_QC

    TEMP_QC_VAR = nc.createVariable('TEMP_QC', 'f4', 'records')
    TEMP_QC_VAR[:] = TEMP_QC

    PRES_QC_VAR = nc.createVariable('PRES_QC', 'f4', 'records')
    PRES_QC_VAR[:] = PRES_QC

    CNDC_QC_VAR = nc.createVariable('CNDC_QC', 'f4', 'records')
    CNDC_QC_VAR[:] = CNDC_QC

    nc.close()

def read_csv_files(input_filepath, dest_filepath, float_num, broken_float):
    """
    Parse raw RBR float CSV/TXT log files and write intermediate netCDF files.

    Reads paired science_log.csv (LGR_CP_PTSCI rows with PRES, TEMP, SAL, CNDC,
    TEMP_CNDC, COUNT) and system_log.txt (GPS timestamp, surface pressure offset)
    files for each profile. Data is reversed to match ARGO convention (increasing
    pressure). Calls make_nc_file_origin() for each profile.

    Parameters
    ----------
    input_filepath : str
        Directory containing science_log.csv and system_log.txt files.
    dest_filepath : str
        Output directory for intermediate netCDF files.
    float_num : str
        Float identifier (e.g. 'F9186').
    broken_float : int
        0 = read from standard LGR_CP_PTSCI records (normal floats).
        1 = read from LGR_PTSCI records (fallback for floats with logging issues).
        Raises Exception if any other value is passed.
    """
    
    all_files = (p.resolve() for p in Path(input_filepath).glob("*") if p.name.endswith("system_log.txt") or p.name.endswith("science_log.csv"))
    
    files_dictionary = {}
    for file_path in all_files:
        filename = file_path.name
        profile_num = filename[9:12]  # Extract the profile number
        if profile_num not in files_dictionary:
            files_dictionary[profile_num] = {}
        if "science_log.csv" in filename:
            files_dictionary[profile_num]["science_log"] = file_path
        elif "system_log.txt" in filename:
            files_dictionary[profile_num]["system_log"] = file_path
            
    # Process each profile only if both files are present
    for profile_num, file_paths in files_dictionary.items():

        if "science_log" in file_paths and "system_log" in file_paths:

            # Initialize variables
            pressures, temps, sals, cndc, temp_cndc, counts, PTSCI_timestamps = [], [], [], [], [], [], []
            latitude, longitude, juld_location, juld_timestamp = None, None, None, None
    
            # Read the science log file
            with open(file_paths["science_log"], mode='r') as sci_file:
                reader = csv.reader(sci_file)
                PTSCIinfo = []
                GPS = []
                JULD_timestamp = None
                prev_line = None
                BROKEN_CP_PTSCI_ASCENT_FLAG = False
                BROKEN_PTSCI_DESCENT_FLAG = False
                for row in reader:
                    # commented out code pertains to "broken" float F10051, where we played around with taking PTSCI data from the descent
                    # if(row[2] == "Park Descent Mission"):
                    #     BROKEN_PTSCI_DESCENT_FLAG = True
                    # if(row[2] == "Park Mission"):
                    #     BROKEN_PTSCI_DESCENT_FLAG = False   
                    if(row[2] == "ASCENT"):
                        BROKEN_CP_PTSCI_ASCENT_FLAG = True
                    # if broken_float == 1:
                    #     if BROKEN_PTSCI_DESCENT_FLAG == True:
                    #         if row[0] == 'LGR_PTSCI':
                    #             # -99 to indicate no vals were avg for this measurement
                    #             row.append(-99)
                    #             PTSCIinfo.append(row)
                    if row[2] == "CP started":
                        JULD_timestamp = row[1]
                    if broken_float == 1:
                        # Different broken float code... 
                        if BROKEN_CP_PTSCI_ASCENT_FLAG == True:
                            if row[0] == 'LGR_PTSCI':
                                # -99 to indicate no vals were avg for this measurement
                                row.append(-99)
                                PTSCIinfo.append(row)
                            if row[0] == 'LGR_CP_PTSCI':
                                PTSCIinfo.append(row)
                            if row[2] == "CP started":
                                JULD_timestamp = row[1]
                    elif broken_float == 0:
                        if(row[0] == "LGR_CP_PTSCI"):
                            PTSCIinfo.append(row)  
                    else:
                        raise Exception("Please enter a valid number for if this is a broken float")
                    
                    # Get lon/ lat vals
                    if(row[0] == 'GPS'):
                        GPS.append(row)
                    if(row[2] == "Surface Mission"):
                        JULD_timestamp = prev_line[1]
                    prev_line = row
                    
            # Process data from science file
            for row in PTSCIinfo:
                PTSCI_timestamps.append(int(row[1].replace('T', '')))
                pressures.append(float(row[2]))
                temps.append(float(row[3]))
                sals.append(float(row[4]))
                cndc.append(float(row[5]))
                temp_cndc.append(float(row[6]))
                counts.append(int(row[-1]))

            try:
                if len(GPS) != 0:
                    GPS = GPS[-1]
                    latitude = round(float(GPS[2]), 4)
                    longitude = round(float(GPS[3]),4)   
                    juld_location = to_julian_day(datetime.strptime(GPS[1], "%Y%m%dT%H%M%S"))
                else:
                    print(f"GPS not present for {profile_num}")
                    latitude = np.nan
                    longitude = np.nan
                    juld_location = np.nan
            except IndexError as e:
                print(f"Invalid Lat/ Lon for {profile_num}")
                latitude = np.nan
                longitude = np.nan
                juld_location = np.nan
            
            if JULD_timestamp is not None:
                juld_timestamp = to_julian_day(datetime.strptime(JULD_timestamp, "%Y%m%dT%H%M%S"))
            else:
                # Use last PTSCI (deepest) measurement as timestamp if it exists
                if len(PTSCI_timestamps) > 0:
                    juld_timestamp = to_julian_day(datetime.strptime(str(PTSCI_timestamps[0]), "%Y%m%d%H%M%S"))
                    print(f"JULD not present for {profile_num}, using last PTSCI timestamp instead")
                else:
                    juld_timestamp = np.nan
                    print(f"JULD not present for {profile_num}")

            # Init vars for sys file
            offset, PSAL_ADJUSTED, TEMP_ADJUSTED, PRES_ADJUSTED  = None, None, None, None
    
            with open(file_paths["system_log"], mode='r') as sys_file:
                for line in sys_file:
                    if 'surface pressure offset' in line:
                        line = line.split(' ')
                        offset = line[-2]

            # Reverse data to be consistent with ARGO NETCDF files
            pressures.reverse()
            temps.reverse()
            sals.reverse()
            cndc.reverse()
            temp_cndc.reverse()
            counts.reverse()
            PTSCI_timestamps.reverse()

            # # bin avg data according to pressure - bin size is 2DBAR
            # this is code to bin avg broken float data for F10051
            # bin_edges = np.arange(np.nanmin(pressures), np.nanmax(pressures) + 2, 2)
            # pres_binned = stats.binned_statistic(pressures, pressures, 'mean', bins=bin_edges).statistic
            # temp_binned = stats.binned_statistic(pressures, temps, 'mean', bins=bin_edges).statistic
            # cndc_binned = stats.binned_statistic(pressures, cndc, 'mean', bins=bin_edges).statistic
            # temp_cndc_binned = stats.binned_statistic(pressures, temp_cndc, 'mean', bins=bin_edges).statistic
            # counts_binned = stats.binned_statistic(pressures, pressures, statistic='count', bins=bin_edges).statistic
            # # practical salinity 
            # psal_binned = gsw.SP_from_C(cndc_binned, temp_binned, pres_binned)

            # # make sure to reverse data after binnning ###
            # pressures = pres_binned
            # temps = temp_binned
            # sals = psal_binned
            # cndc = cndc_binned
            # temp_cndc = temp_cndc_binned
            # counts = counts_binned

            PTSCI_timestamps = np.asarray(PTSCI_timestamps)
            TEMP_ADJUSTED =  np.asarray(copy.deepcopy(temps))
            PSAL_ADJUSTED =  np.asarray(copy.deepcopy(sals))
            cndc = np.asarray(cndc)/10

            # init VAR_ADJUSTED arrs
            if offset is None:
                print(f"Profile {profile_num} is missing 'surface pressure offset' in system log")
                print("[VAR]_ADJUSTED arrays will be initalized as copies of orginal data arrs, no pressure offset applied")
                PRES_ADJUSTED = np.asarray(copy.deepcopy(pressures))
                make_nc_file_origin(profile_num, pressures, temps, sals, cndc, temp_cndc, counts,
                                    PRES_ADJUSTED, TEMP_ADJUSTED, PSAL_ADJUSTED,
                                    latitude, longitude, juld_timestamp, juld_location, dest_filepath, float_num,
                                    pres_offset=offset, PTSCI_TIMESTAMPS=PTSCI_timestamps)
            else:
                PRES_ADJUSTED =  np.asarray(copy.deepcopy(pressures)) - float(offset)
                make_nc_file_origin(profile_num, pressures, temps, sals, cndc, temp_cndc, counts,
                                    PRES_ADJUSTED, TEMP_ADJUSTED, PSAL_ADJUSTED,
                                    latitude, longitude, juld_timestamp, juld_location, dest_filepath, float_num,
                                    pres_offset=offset, PTSCI_TIMESTAMPS=PTSCI_timestamps)

        else:
           print(f"Skipping profile {profile_num}: Missing required files.")

def read_argo_nc_files(nc_filepath, dest_filepath, float_num):
    """
    Read real-time ARGO netCDF profile files and write intermediate netCDF files.

    Reads all *.nc files in nc_filepath, extracts standard ARGO variables (PRES,
    TEMP, PSAL, CNDC, TEMP_CNDC, NB_SAMPLE_CTD, JULD, JULD_LOCATION, LAT, LON),
    and pre-existing QC flags. QC arrays that are missing, single-valued, or empty
    are defaulted to 0 (no QC). Calls make_nc_file_origin() for each profile.

    Parameters
    ----------
    nc_filepath : str
        Directory containing real-time ARGO *.nc profile files.
    dest_filepath : str
        Output directory for intermediate netCDF files.
    float_num : str
        Float identifier (e.g. '1902655').
    """
    
    nc_files = glob.glob(os.path.join(nc_filepath, "*.nc"))
   
    for file in nc_files:

        nc = nc4.Dataset(file)
        
        profile_num = int(os.path.basename(file).split("_")[1].split(".nc")[0])
        pressures = np.squeeze(nc.variables['PRES'][:].filled(np.NaN))
        temps = np.squeeze(nc.variables['TEMP'][:].filled(np.NaN))
        sals = np.squeeze(nc.variables['PSAL'][:].filled(np.NaN))
        cndc = np.squeeze(nc.variables['CNDC'][:].filled(np.NaN))
        temp_cndc = np.squeeze(nc.variables['TEMP_CNDC'][:].filled(np.NaN))
        counts = nc.variables['NB_SAMPLE_CTD'][:].filled(-99)
        latitude = nc.variables['LATITUDE'][:].filled(np.NaN)
        longitude = nc.variables['LONGITUDE'][:].filled(np.NaN)
        juld_timestamp = nc.variables['JULD'][:].filled(np.NaN)
        juld_location = nc.variables['JULD_LOCATION'][:].filled(np.NaN)
        
        PSAL_ADJUSTED = np.squeeze(nc.variables['PSAL_ADJUSTED'][:].filled(np.NaN))
        psal_adjusted_qc_temp = np.squeeze(nc.variables['PSAL_ADJUSTED_QC'][:].filled(np.NaN))
        if psal_adjusted_qc_temp.size != 1 and psal_adjusted_qc_temp.size != 0:
            PSAL_ADJUSTED_QC = np.asarray([int(x) for x in psal_adjusted_qc_temp])
        else:
            PSAL_ADJUSTED_QC = np.full(sals.shape, fill_value=0)
        psal_qc_temp = np.squeeze(nc.variables['PSAL_QC'][:].filled(np.NaN))
        if psal_qc_temp.size != 1 and psal_qc_temp.size != 0:
            PSAL_QC = np.asarray([int(x) for x in psal_qc_temp])
        else:
            PSAL_QC = np.full(sals.shape, fill_value=0)

        TEMP_ADJUSTED = np.squeeze(nc.variables['TEMP_ADJUSTED'][:].filled(np.NaN))
        temp_adjusted_qc_temp = np.squeeze(nc.variables['TEMP_ADJUSTED_QC'][:].filled(np.NaN))
        if temp_adjusted_qc_temp.size != 1 and temp_adjusted_qc_temp.size != 0:
            TEMP_ADJUSTED_QC = np.asarray([int(x) for x in temp_adjusted_qc_temp])
        else:
            TEMP_ADJUSTED_QC = np.full(temps.shape, fill_value=0)
        temp_qc_temp = np.squeeze(nc.variables['TEMP_QC'][:].filled(np.NaN))
        if temp_qc_temp.size != 1 and temp_qc_temp.size != 0:
            TEMP_QC = np.asarray([int(x) for x in temp_qc_temp])
        else:
            TEMP_QC = np.full(temps.shape, fill_value=0)

        PRES_ADJUSTED = np.squeeze(nc.variables['PRES_ADJUSTED'][:].filled(np.NaN))
        pres_adjusted_qc_temp = np.squeeze(nc.variables['PRES_ADJUSTED_QC'][:].filled(np.NaN))
        if pres_adjusted_qc_temp.size != 1 and pres_adjusted_qc_temp.size != 0:
            PRES_ADJUSTED_QC = np.asarray([int(x) for x in pres_adjusted_qc_temp])
        else:
            PRES_ADJUSTED_QC = np.full(pressures.shape, fill_value=0)
        pres_qc_temp = np.squeeze(nc.variables['PRES_QC'][:].filled(np.NaN))
        if pres_qc_temp.size != 1 and pres_qc_temp.size != 0:
            PRES_QC = np.asarray([int(x) for x in pres_qc_temp])
        else:
            PRES_QC = np.full(pressures.shape, fill_value=0)

        
        cndc_qc_temp = np.squeeze(nc.variables['CNDC_QC'][:].filled(np.NaN))
        if cndc_qc_temp.size != 1 and cndc_qc_temp.size != 0:
            CNDC_QC = np.asarray([int(x) for x in cndc_qc_temp])
        else:
            CNDC_QC = np.full(pressures.shape, fill_value=0)

        POSITION_QC = int(nc.variables['POSITION_QC'][:].filled(np.NaN)[0])
        JULD_QC = int(nc.variables['JULD_QC'][:].filled(np.NaN)[0])
        
        temperature_cndc_qc_temp = np.squeeze(nc.variables['TEMP_CNDC_QC'][:].filled(np.NaN))
        if temperature_cndc_qc_temp.size != 1 and temperature_cndc_qc_temp.size != 0:
            TEMP_CNDC_QC = np.asarray([int(x) for x in temperature_cndc_qc_temp])
        else:
            TEMP_CNDC_QC = np.full(pressures.shape, fill_value=0)
        
        nc_sample_ctd_qc_temp = np.squeeze(nc.variables['NB_SAMPLE_CTD_QC'][:].filled(np.NaN))
        if nc_sample_ctd_qc_temp.size != 1 and nc_sample_ctd_qc_temp.size != 0:
            NB_SAMPLE_CTD_QC = np.asarray([int(x) for x in nc_sample_ctd_qc_temp])
        else:
            NB_SAMPLE_CTD_QC = np.full(pressures.shape, fill_value=0)
        
        make_nc_file_origin(profile_num, pressures, temps, sals, cndc, temp_cndc, counts,
                            PRES_ADJUSTED, TEMP_ADJUSTED, PSAL_ADJUSTED,
                            latitude, longitude, juld_timestamp, juld_location, dest_filepath, float_num,
                            PSAL_QC = PSAL_QC,
                            TEMP_QC = TEMP_QC,
                            PRES_QC = PRES_QC,
                            CNDC_QC = CNDC_QC,
                            PSAL_ADJUSTED_QC=PSAL_ADJUSTED_QC, 
                            TEMP_ADJUSTED_QC=TEMP_ADJUSTED_QC,
                            PRES_ADJUSTED_QC=PRES_ADJUSTED_QC,
                            POSITION_QC=POSITION_QC,
                            JULD_QC=JULD_QC,
                            TEMP_CNDC_QC=TEMP_CNDC_QC,
                            NB_SAMPLE_CTD_QC=NB_SAMPLE_CTD_QC)

def download_files(url, download_dir, float_num):
    """
    Download all profile files for a float from an ARGO data center URL.

    Fetches the directory listing at url using BeautifulSoup, filters for links
    containing the float number with R or D prefix, then downloads all matching
    files in parallel using ProcessPoolExecutor.

    Parameters
    ----------
    url : str
        ARGO data center URL listing float profiles
        (e.g. 'https://data-argo.ifremer.fr/dac/aoml/1902655/profiles/').
    download_dir : str
        Local directory where downloaded files are saved.
    float_num : str or int
        Float number used to filter links (e.g. '1902655').
    """

    # Get links
    res = requests.get(url)
    data = res.text
    soup = BeautifulSoup(data)

    download_tasks = []

    for link in soup.find_all('a'):
        if (f"R{float_num}" or f"D{float_num}") in link.get('href'):
            with ProcessPoolExecutor() as executor:
                try:
                    full_dload_link = f"{url}{link.get('href')}"
                    download_tasks.append(executor.submit(download_file, full_dload_link, download_dir, link.get('href')))
                except Exception as e:
                    print(e)

    # Wait for all tasks to complete
    for task in download_tasks:
        try:
            task.result() # This will raise any exceptions that occurred in the task
        except Exception as e:
            print(e)

    print("Finished Downloading Profiles")

def download_file(dload_url, download_dir, float_num):
    """
    Download a single file from a URL and save it to download_dir.

    Used internally by download_files() via ProcessPoolExecutor. Checks for HTTP
    200 status and writes the binary response content to disk.

    Parameters
    ----------
    dload_url : str
        Full URL of the file to download.
    download_dir : str
        Local directory where the file is saved.
    float_num : str or int
        Float number (used only in the print statement, not in the filename).
    """

    response = requests.get(dload_url, timeout=5)
    if response.status_code == 200:
        with open(os.path.join(download_dir, float_num), "wb") as dload_file:
            dload_file.write(response.content)
    else:
        print("Failed to download file")

def main():

    download_ARGO_NETCDF_files = 0
    dload_url = "https://data-argo.ifremer.fr/dac/aoml/6990591/profiles/"
    read_ARGO_NETCDF_files = 1
    read_RAW_CSV_files = 0

    float_num = "7902322"
    input_dir = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F9443\F9443_new")
    dest_filepath = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F9443\F9443_new_0")

    if download_ARGO_NETCDF_files == 1:
        argo_internal_float_num = "6990591"
        download_files(dload_url, input_dir, argo_internal_float_num)

    if read_ARGO_NETCDF_files == 1:
        read_argo_nc_files(input_dir, dest_filepath, float_num)

    if read_RAW_CSV_files == 1:
        broken_float = 0
        read_csv_files(input_dir, dest_filepath, float_num, broken_float)
    
if __name__ == '__main__':

    main()
