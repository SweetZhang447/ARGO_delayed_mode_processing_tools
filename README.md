# ARGO Delayed Mode Processing Tools
This repository provides a set of Python-based tools for the delayed mode processing of ARGO float data.
Below includes instructions for various scripts needed to run the pipeline, in order. 

DRAFT VERSION 4/17/2025
Please email me at xuanqin.zhang@sjsu.edu if you have any questions or comments!

## make_origin_nc_files.py
This script converts either raw float .csv files or real time ARGO NetCDF files into a intermediate NetCDF format for delayed mode processing.
Inside of the main module, there are 7 input parameters to specify.

FLAG          | DESCRIPTION
------------- | -------------
download_ARGO_NETCDF_files | 1 to download a set of real time ARGO NETCDF files, 0 otherwise
dload_url                  | URL to download real time ARGO NETCDF files from; EX: "https://data-argo.ifremer.fr/dac/aoml/1902655/profiles/"
argo_internal_float_num    | The number in front of the profile number for ARGO real time files; EX: 1902655
read_ARGO_NETCDF_files     | 1 to read a set of real time ARGO NETCDF files, 0 otherwise
read_RAW_CSV_files         | 1 to read a set of .csv float files, 0 otherwise
float_num                  | Float ID; EX: F9186
input_dir                  | Input directory
dest_filepath              | Output directory 

## delayed_mode_processing.py
This scripts reads the intermediate NETCDF files generated from the previous step, and provides a variety of tools and functions for the flagging and verification of data.

FLAG          | DESCRIPTION
------------- | -------------
float_num                  | Float ID; EX: F9186
nc_filepath                | Path to generated intermediate NETCDF files
dest_filepath              | Output directory 

Some functions may require more input parameters, more details outlined below.

### FUNCTION: first_time_run(nc_filepath, dest_filepath, float_num)
This function is designed to be the first step run after the generation of the intermediate NETCDF files.
Steps and checks are outlined below:

1. verify_autoset_qc_flags()
   - If your intermediate file was generated from real time ARGO NETCDF files, then there are already QC flags present. This module pops up a series of graphs so the user can verify the validity of these autoset flags.
2. juld_check()
   - Checks for missing JULD data values
   - If JULD_LOCATION is available, then fill in JULD with that value, set JULD_QC to 5 to indicate changed value
   - If both are missing, then interpolate values for JULD and set JULD_QC to 8 to indicate interpolated value
3. lat_lon_check()
   - Checks for missing LAT/ LON values
   - If missing, then interpolates LAT/LON and sets POSITION_QC to 8 
4. count_check()
   - For bin averaged data, if (NB_SAMPLE_CTD > 50) OR (NB_SAMPLE_CTD < 1 AND NB_SAMPLE_CTD != -99), then set PSAL, CNDC, and TEMP ADJUSTED_QC arrays to 3 to indicate "probably bad" value.
   - NOTE: -99 is used to indicate misssing/ nonexistent NB_SAMPLE_CTD values.
5.  pres_depth_check()
   - Checks where pressure < 1 dbar, and marks PSAL and CNDC ADJUSTED_QC as 4 to indicate bad value.
6. temp_check()
   - Checks where pressure < 0.1 dbar, and where TEMP < -2, sets TEMP_ADJUSTED_QC as 4 to indicate bad value.

### FUNCTION: manipulate_data_flags(nc_filepath, dest_filepath, float_num, profile_number)
This function is designed to be used to look at and set QC flags for a single profile. 
There are various functions within this module that you can comment in or out depending on your needs.

FLAG          | DESCRIPTION
------------- | -------------
profile_number                      | profile number to look in depth at 
PARAM (not needed in all functions) | pass in either: PRES, PSAL, or TEMP as a string

1. data_snapshot_graph(argo_data, profile_number)
   - Generates a graph containing: TS, PRES v PSAL, and PRES v TEMP, along with LAT/LON and JULD information for corresponding profile
   - Ability to flag points for PSAL and TEMP, click on the same point to cycle through QC flag options (good, probably good, probably bad, bad)
2. flag_data_points(argo_data, profile_number, PARAM)
  - For PARAM, pass in either: PRES, PSAL, or TEMP as a string
    - Generates PRESSURE v COUNT, PRES v PSAL, or PRES v TEMP graph
  - Ability to flag individual points, click on the same point to cycle through QC flag options (good, probably good, probably bad, bad)
3. flag_range_data(argo_data, profile_number, PARAM)
  - Ability to flag multiple ranges of points, click on the same point to cycle through QC flag options (good, probably good, probably bad, bad), then a secondary point to define a range.
4. flag_TS_data(argo_data, profile_num)
  - Ability to flag PSAL or TEMP points in TS graph
  - Cycles through option to mark both TEMP and PSAL as good, PSAL as bad and TEMP as good, TEMP as bad and PSAL as good, or both as bad.
  - Only able to mark points as good or bad, no probably good/bad options availble

### FUNCTION: generate_dataset_graphs(nc_filepath, dest_filepath, float_num, qc_arr_selection, data_type, use_adjusted)
This function is designed to look at a complete dataset, then pick out from that dataset profiles of interest. 

FLAG          | DESCRIPTION
------------- | -------------
qc_arr_selection           | Array of QC values to look at; EX: if you only want good/ probably good data, pass in [0, 1, 2]
data_type                  | Either "PSAL" or "TEMP"
use_adjusted               | True if you want to use PARAM_ADJUSTED data and associated PARAM_ADJUSTED_QC arrays, otherwise false

1. graph_pres_v_var_all()
   - Generates a PRES v DATA_TYPE graph for all profiles
   - Ability to click and select profiles of interest
2. graph_TS_all
   - Generates a TS graph for all profiles
   - Ability to click and select profiles of interest
3. graph_deep_section_var_all
   - Generates a DEEP SECTION graph for all profiles

After exiting out of the aforementioned graphs, selected profiles of interest will pop out a data_snapshot_graph, and flag_range_data_graph for TEMP and PSAL where you can manipulate QC flags

## make_final_nc_files.py
This module makes the final delayed mode ARGO NETCDF files. 

FLAG          | DESCRIPTION
------------- | -------------
float_num                 | Float number
dest_filepath             | Output directory 
nc_filepath               | Input directory of intermediate NETCDF files
orgargo_netcdf_filepath (OPT)  | Directory for original argo NETCDF files, if it exists
config_fp                 | Filepath to config file

## make_config_file(float_num, dest_filepath, org_argo_netcdf_filepath = orgargo_netcdf_filepath)
Pass in orgargo_netcdf_filepath to make config file for params needed to generate delayed mode NETCDF files. 
Otherwise leave empty and this function will generate a empty text file for user to manually fill out. 

## process_data_dmode_files(nc_filepath, float_num, dest_filepath, config_fp, org_netcdf_fp = orgargo_netcdf_filepath)
Pass in orgargo_netcdf_filepath, if it exists, in order to generate delayed mode files from the original real time ARGO files. 
For example, if passed in, parameters like "date_creation" will be taken from the original real-time files, new scientific_calib and history comments will be appended to existing ones. 


