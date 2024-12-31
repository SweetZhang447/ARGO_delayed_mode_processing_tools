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

def make_nc_file_origin(profile_num, pressures, temps, sals, counts,
                        PRES_ADJUSTED, TEMP_ADJUSTED, PSAL_ADJUSTED,
                        latitude, longitude, juld_timestamp, juld_location, dest_filepath, float_num,
                        **kwargs):

    output_filename = os.path.join(dest_filepath, f"{float_num}-{profile_num:03}.nc")
    nc = nc4.Dataset(output_filename, 'w')

    # Set global attributes
    # TODO: make more detailed later
    nc.author = 'Sweet Zhang'
    nc.summary = 'NC file of CSV: LGR_CP_PTSCI readings ONLY'

    # convert all lists to numpy array types
    pressures = np.asarray(pressures)
    temps = np.asarray(temps)
    sals = np.asarray(sals)
    counts = np.asarray(counts)
    
    # Initialize variables with values passed through kwargs, or use defaults if not provided
    PSAL_ADJUSTED_ERROR = kwargs.get("PSAL_ADJUSTED_ERROR", np.full(sals.shape, np.nan))
    PSAL_ADJUSTED_QC = kwargs.get("PSAL_ADJUSTED_QC", np.full(sals.shape, fill_value=0))

    TEMP_ADJUSTED_ERROR = kwargs.get("TEMP_ADJUSTED_ERROR", np.full(temps.shape, np.nan))
    TEMP_ADJUSTED_QC = kwargs.get("TEMP_ADJUSTED_QC", np.full(temps.shape, fill_value=0))

    PRES_ADJUSTED_ERROR = kwargs.get("PRES_ADJUSTED_ERROR", np.full(pressures.shape, np.nan))
    PRES_ADJUSTED_QC = kwargs.get("PRES_ADJUSTED_QC", np.full(pressures.shape, fill_value=0))

    CNDC_ADJUSTED_QC = kwargs.get("CNDC_ADJUSTED_QC", np.full(pressures.shape, fill_value=0)) 

    PSAL_QC = kwargs.get("PSAL_QC", np.full(pressures.shape, fill_value=0)) 
    TEMP_QC = kwargs.get("TEMP_QC", np.full(pressures.shape, fill_value=0)) 
    PRES_QC = kwargs.get("PRES_QC", np.full(pressures.shape, fill_value=0)) 
    CNDC_QC = kwargs.get("CNDC_QC", np.full(pressures.shape, fill_value=0)) 

    POSITION_QC = kwargs.get("POSITION_QC", np.nan)
    JULD_QC = kwargs.get("JULD_QC", np.nan)

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

    counts_var = nc.createVariable('COUNTS', 'f4', 'records')
    counts_var[:] = counts

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

    PSAL_ADJUSTED_ERROR_VAR = nc.createVariable('PSAL_ADJUSTED_ERROR', 'f4', 'records')
    PSAL_ADJUSTED_ERROR_VAR[:] = PSAL_ADJUSTED_ERROR

    PSAL_ADJUSTED_QC_VAR = nc.createVariable('PSAL_ADJUSTED_QC', 'f4', 'records')
    PSAL_ADJUSTED_QC_VAR[:] = PSAL_ADJUSTED_QC

    TEMP_ADJUSTED_VAR = nc.createVariable('TEMP_ADJUSTED', 'f4', 'records')
    TEMP_ADJUSTED_VAR[:] = TEMP_ADJUSTED

    TEMP_ADJUSTED_ERROR_VAR = nc.createVariable('TEMP_ADJUSTED_ERROR', 'f4', 'records')
    TEMP_ADJUSTED_ERROR_VAR[:] = TEMP_ADJUSTED_ERROR

    TEMP_ADJUSTED_QC_VAR = nc.createVariable('TEMP_ADJUSTED_QC', 'f4', 'records')
    TEMP_ADJUSTED_QC_VAR[:] = TEMP_ADJUSTED_QC

    PRES_ADJUSTED_VAR = nc.createVariable('PRES_ADJUSTED', 'f4', 'records')
    PRES_ADJUSTED_VAR[:] = PRES_ADJUSTED

    PRES_ADJUSTED_ERROR_VAR = nc.createVariable('PRES_ADJUSTED_ERROR', 'f4', 'records')
    PRES_ADJUSTED_ERROR_VAR[:] = PRES_ADJUSTED_ERROR

    PRES_ADJUSTED_QC_VAR = nc.createVariable('PRES_ADJUSTED_QC', 'f4', 'records')
    PRES_ADJUSTED_QC_VAR[:] = PRES_ADJUSTED_QC

    CNDC_ADJUSTED_QC_VAR = nc.createVariable('CNDC_ADJUSTED_QC', 'f4', 'records')
    CNDC_ADJUSTED_QC_VAR[:] = CNDC_ADJUSTED_QC

    PSAL_QC_VAR = nc.createVariable('PSAL_QC', 'f4', 'records')
    PSAL_QC_VAR[:] = PSAL_QC

    TEMP_QC_VAR = nc.createVariable('TEMP_QC', 'f4', 'records')
    TEMP_QC_VAR[:] = TEMP_QC

    PRES_QC_VAR = nc.createVariable('PRES_QC', 'f4', 'records')
    PRES_QC_VAR[:] = PRES_QC

    CNDC_QC_VAR = nc.createVariable('CNDC_QC', 'f4', 'records')
    CNDC_QC_VAR[:] = CNDC_QC

    QC_FLAG_CHECK_VAR = nc.createVariable('QC_FLAG_CHECK', 'f4', 'single_record')
    QC_FLAG_CHECK_VAR[:] = 0

    nc.close()

def read_csv_files(csv_filepath, dest_filepath, float_num, broken_float):

    
    all_files = (p.resolve() for p in Path(csv_filepath).glob("*") if p.name.endswith("system_log.txt") or p.name.endswith("science_log.csv"))
    
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
            pressures, temps, sals, counts = [], [], [], []
            latitude, longitude, juld_location, juld_timestamp = None, None, None, None
    
            # Read the science log file
            with open(file_paths["science_log"], mode='r') as sci_file:
                reader = csv.reader(sci_file)
                PTSCIinfo = []
                GPS = []
                JULD_timestamp = None
                prev_line = None
                BROKEN_CP_PTSCI_ASCENT_FLAG = False
                for row in reader:
                    if(row[2] == "ASCENT"):
                        BROKEN_CP_PTSCI_ASCENT_FLAG = True
                    if broken_float == 1:
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
                pressures.append(float(row[2]))
                temps.append(float(row[3]))
                sals.append(float(row[4]))
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
                juld_timestamp = np.nan
                print(f"JULD not present for {profile_num}")

            # Init vars for sys file
            offset, PSAL_ADJUSTED, TEMP_ADJUSTED, PRES_ADJUSTED = None, None, None, None
    
            with open(file_paths["system_log"], mode='r') as sys_file:
                for line in sys_file:
                    if 'surface pressure offset' in line:
                        line = line.split(' ')
                        offset = line[-2]

            # init VAR_ADJUSTED arrs
            if offset is None:
                print(f"Profile {profile_num} is missing 'surface pressure offset' in system log")
                print("[VAR]_ADJUSTED arrays will be initalized as copies of orginal data arrs, no pressure offset applied")
                PRES_ADJUSTED = np.asarray(copy.deepcopy(pressures))
                TEMP_ADJUSTED = np.asarray(copy.deepcopy(temps))
                PSAL_ADJUSTED = np.asarray(copy.deepcopy(sals))
                make_nc_file_origin(profile_num, pressures, temps, sals, counts,
                                    PRES_ADJUSTED, TEMP_ADJUSTED, PSAL_ADJUSTED,
                                    latitude, longitude, juld_timestamp, juld_location, dest_filepath, float_num)
            else:
                PRES_ADJUSTED =  np.asarray(copy.deepcopy(pressures)) - float(offset)
                TEMP_ADJUSTED =  np.asarray(copy.deepcopy(temps))
                PSAL_ADJUSTED =  np.asarray(copy.deepcopy(sals))
                make_nc_file_origin(profile_num, pressures, temps, sals, counts,
                                    PRES_ADJUSTED, TEMP_ADJUSTED, PSAL_ADJUSTED,
                                    latitude, longitude, juld_timestamp, juld_location, dest_filepath, float_num)

        
        else:
           print(f"Skipping profile {profile_num}: Missing required files.")

def read_argo_nc_files(nc_filepath, dest_filepath, float_num):
    
    nc_files = glob.glob(os.path.join(nc_filepath, "*.nc"))
   
    for file in nc_files:

        nc = nc4.Dataset(file)
        
        profile_num = int(os.path.basename(file).split("_")[1].split(".nc")[0])
        pressures = np.squeeze(nc.variables['PRES'][:].filled(np.NaN))
        temps = np.squeeze(nc.variables['TEMP'][:].filled(np.NaN))
        sals = np.squeeze(nc.variables['PSAL'][:].filled(np.NaN))
        counts = nc.variables['NB_SAMPLE_CTD'][:].filled(-99)
        latitude = nc.variables['LATITUDE'][:].filled(np.NaN)
        longitude = nc.variables['LONGITUDE'][:].filled(np.NaN)
        juld_timestamp = nc.variables['JULD'][:].filled(np.NaN)
        juld_location = nc.variables['JULD_LOCATION'][:].filled(np.NaN)
        
        PSAL_ADJUSTED = np.squeeze(nc.variables['PSAL_ADJUSTED'][:].filled(np.NaN))
        PSAL_ADJUSTED_ERROR = np.squeeze(nc.variables['PSAL_ADJUSTED_ERROR'][:].filled(np.NaN))
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
        TEMP_ADJUSTED_ERROR = np.squeeze(nc.variables['TEMP_ADJUSTED_ERROR'][:].filled(np.NaN))
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
        PRES_ADJUSTED_ERROR = np.squeeze(nc.variables['PRES_ADJUSTED_ERROR'][:].filled(np.NaN))
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

        cndc_adjusted_qc_temp = np.squeeze(nc.variables['CNDC_ADJUSTED_QC'][:].filled(np.NaN))
        if cndc_adjusted_qc_temp.size != 1 and cndc_adjusted_qc_temp.size != 0:
            CNDC_ADJUSTED_QC = np.asarray([int(x) for x in cndc_adjusted_qc_temp])
        else:
            CNDC_ADJUSTED_QC = np.full(pressures.shape, fill_value=0)
        cndc_qc_temp = np.squeeze(nc.variables['CNDC_QC'][:].filled(np.NaN))
        if cndc_qc_temp.size != 1 and cndc_qc_temp.size != 0:
            CNDC_QC = np.asarray([int(x) for x in cndc_qc_temp])
        else:
            CNDC_QC = np.full(pressures.shape, fill_value=0)

        POSITION_QC = int(nc.variables['POSITION_QC'][:].filled(np.NaN)[0])
        JULD_QC = int(nc.variables['JULD_QC'][:].filled(np.NaN)[0])

        make_nc_file_origin(profile_num, pressures, temps, sals, counts,
                            PRES_ADJUSTED, TEMP_ADJUSTED, PSAL_ADJUSTED,
                            latitude, longitude, juld_timestamp, juld_location, dest_filepath, float_num,
                            PSAL_QC = PSAL_QC,
                            TEMP_QC = TEMP_QC,
                            PRES_QC = PRES_QC,
                            CNDC_QC = CNDC_QC,
                            PSAL_ADJUSTED_ERROR=PSAL_ADJUSTED_ERROR,
                            PSAL_ADJUSTED_QC=PSAL_ADJUSTED_QC,
                            TEMP_ADJUSTED_ERROR=TEMP_ADJUSTED_ERROR,
                            TEMP_ADJUSTED_QC=TEMP_ADJUSTED_QC,
                            PRES_ADJUSTED_ERROR=PRES_ADJUSTED_ERROR,
                            PRES_ADJUSTED_QC=PRES_ADJUSTED_QC,
                            CNDC_ADJUSTED_QC=CNDC_ADJUSTED_QC,
                            POSITION_QC=POSITION_QC,
                            JULD_QC=JULD_QC)

def download_files(url, download_dir, float_num):

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

def download_file(dload_url, download_dir, curr_filename):

    response = requests.get(dload_url, timeout=5)
    if response.status_code == 200:
        with open(os.path.join(download_dir, curr_filename), "wb") as dload_file:
            dload_file.write(response.content)
    else:
        print("Failed to download file")

def main(download_ARGO_NETCDF_files, read_ARGO_NETCDF_files, read_RAW_CSV_files):

    float_num= "F10051"
    # F10051_bad_data_30_69        F10051_all_data
    # input_dir = "C:\\Users\\szswe\\Desktop\\NOAA_pipeline\\F10051_data\\F10051_bad_data_30_69"
    # dest_filepath = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\csv_to_nc\\F10051_0"
    input_dir = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\RAW_DATA\\F10051_ARGO_NETCDF"
    dest_filepath = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\argo_to_nc\\F10051_0"
    # input_dir = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\RAW_DATA\\F9186_raw_csv"
    # dest_filepath = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\csv_to_nc\\F9186_0"

    if download_ARGO_NETCDF_files == 1:
        dload_url = "https://data-argo.ifremer.fr/dac/aoml/1902655/profiles/"
        argo_internal_float_num = "1902655"
        download_files(dload_url, input_dir, argo_internal_float_num)

    if read_ARGO_NETCDF_files == 1:
        read_argo_nc_files(input_dir, dest_filepath, float_num)

    if read_RAW_CSV_files == 1:
        broken_float = 0
        read_csv_files(input_dir, dest_filepath, float_num, broken_float)
    
if __name__ == '__main__':

    download_ARGO_NETCDF_files = 0
    read_ARGO_NETCDF_files = 1
    read_RAW_CSV_files = 0

    main(download_ARGO_NETCDF_files, read_ARGO_NETCDF_files, read_RAW_CSV_files)
