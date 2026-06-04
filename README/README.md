# ARGO Delayed Mode Processing Tools

## Pipeline Overview

Step | Script | Role
---- | ------ | ----
1 | `make_origin_nc_files.py` | Ingest raw CSV logs or real-time ARGO netCDF → intermediate netCDF
2 | `delayed_mode_processing.py` | Automated QC checks + interactive flagging on intermediate files
3 | `make_final_nc_files.py` | Convert intermediate + config file → final ARGO delayed-mode netCDF
(opt) | `make_KML_files/make_kml.py` | Save profile snapshot images and generate a Google Earth KML file

Supporting modules/ scripts:
- `tools.py` — shared utilities (Julian day conversion, intermediate netCDF I/O)
- `graphs_nc.py` — all matplotlib graphs (called by delayed_mode_processing.py)
- `drift_analysis.py` — salinity drift analysis vs reference datasets (separate from workflow)

Please email me at xuanqin.zhang@sjsu.edu if you have any questions or comments!

## Configuration Files

Scripts 2 and 3 are each configured via a TOML file in the same directory. Run either script directly from the command line — no arguments needed:

```
python delayed_mode_processing.py   # reads delayed_mode_config.toml
python make_final_nc_files.py       # reads make_final_nc_config.toml
```

**TOML syntax notes:**
- Booleans: `true` / `false` (lowercase)
- Lists: `[0, 1, 2]`
- Empty/no value: `""` (empty string)
- Windows paths: use single quotes, e.g. `'c:\Users\...\F9186_VI'`
- Sub-sections use dot notation: `[delayed_mode_processing.generate_graphs]`

### `delayed_mode_config.toml`

Set exactly one of the three `run_*` flags to `true` at a time.

```toml
[delayed_mode_processing]
float_num     = "2904018"
nc_filepath   = 'c:\path\to\intermediate\netcdf\dir'
dest_filepath = 'c:\path\to\output\dir'

run_first_time       = false   # run once after make_origin_nc_files
run_manipulate_flags = false   # interactive QC flagging for a single profile
run_generate_graphs  = true    # overview graphs for full dataset

[delayed_mode_processing.manipulate_flags]
profile_num      = 81
flag_points_pres = false
flag_points_psal = false
flag_points_temp = false
flag_range_pres  = false
flag_range_psal  = true
flag_range_temp  = true
flag_ts          = false

[delayed_mode_processing.generate_graphs]
qc_arr_selection  = [0, 1, 2]
data_type         = "PSAL"        # "PSAL" or "TEMP"
use_adjusted      = true
prof_num_filter   = "347-378"     # "" for all profiles; format "START-END"
date_filter_start = ""            # "" for none; format "YYYY_MM_DD_HH_MM_SS"
date_filter_end   = ""
```

### `make_final_nc_config.toml`

```toml
[make_final_nc_files]
float_num               = "1902655"
nc_filepath             = 'c:\path\to\intermediate\netcdf\dir'
dest_filepath           = 'c:\path\to\dmode\output\dir'
argo_config_fp          = 'c:\path\to\float_config_file.txt'
orgargo_netcdf_filepath = ""       # "" to skip; path to original real-time ARGO netCDF files
run_make_config_file    = false    # set true once to generate config file template, then back to false

[make_final_nc_files.psal_offset]
apply = true
value = 0.025                      # PSU offset added to PSAL_ADJUSTED

[make_final_nc_files.sci_calib]
# Strings written to SCIENTIFIC_CALIB_* in output netCDF. "" to omit.
psal_coefficient = "PSAL_OFFSET=+0.025 PSU"
psal_comment     = "..."
psal_equation    = "PSAL_ADJUSTED = PSAL + 0.025"
temp_coefficient = ""
temp_comment     = ""
temp_equation    = ""
cndc_coefficient = ""
cndc_comment     = ""
cndc_equation    = ""
```

## Intermediate netCDF Format

All files between pipeline steps use a custom intermediate format (one .nc file per profile):

Dimension      | Description
-------------- | -----------
`records`       | Depth levels (trailing NaN levels stripped on write) --> Arrays
`single_record` | Profile-level scalars (position, time, offsets)      --> single values

Key variables (records dim): PRES, TEMP, PSAL, CNDC, TEMP_CNDC + _ADJUSTED + _QC variants, NB_SAMPLE_CTD, PTSCI_TIMESTAMPS
Key variables (single_record dim): JULD, JULD_LOCATION, LAT, LON, POSITION_QC, JULD_QC, PRES_OFFSET, PROFILE_NUM

QC flag values: 0=no QC, 1=good, 2=prob good, 3=prob bad, 4=bad, 5=changed, 8=interpolated
Julian days are referenced to 1950-01-01 00:00:00 UTC.
This repository provides a set of Python-based tools for the delayed mode processing of ARGO float data.
Below includes instructions for various scripts needed to run the pipeline, in order. Please specify/ change needed params 
in the "main" function of each file. 

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
This script reads the intermediate netCDF files generated from the previous step and provides tools for QC flagging and verification. Configure it via `delayed_mode_config.toml` (see **Configuration Files** above), then run:

```
python delayed_mode_processing.py
```

Some functions require additional parameters; details below.

### FUNCTION: first_time_run(nc_filepath, dest_filepath, float_num)
This function is designed to be the first step run after the generation of the intermediate NETCDF files.
Steps and checks are outlined below:

1. juld_check()
   - Checks for missing JULD data values
   - If JULD_LOCATION is available, then fill in JULD with that value, set JULD_QC to 5 to indicate changed value
   - If both are missing, then interpolate values for JULD and set JULD_QC to 8 to indicate interpolated value
     - Sets JULD_LOCATION with interpolated JULD value
2. lat_lon_check()
   - Checks for missing LAT/ LON values
   - If missing, then interpolates LAT/LON and sets POSITION_QC to 8
3. verify_autoset_qc_flags_and_density_inversions()
   - If your intermediate file was generated from real time ARGO NETCDF files, then there are already QC flags present. This module pops up a series of graphs so the user can verify the validity of these autoset flags.
   - If any density inversions are found on the profile, it will pop up a datasnapshot graph.
     - Please look in "README" folder for more details regarding this test
4. count_check()
   - For profiles with suspicious counts (NB_SAMPLE_CTD > 100, or < 1, but NOT -99):
    1) If there is a spike in the PSAL values of more than 15 PSU, mark as bad
    2) For the last point in each profile, checks if PSAL difference is larger than 0.02 PSU, if so -> mark as bad
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
   - Salinity density inversions are marked with edge color "fuchsia"
2. flag_data_points(argo_data, profile_number, PARAM)
  - For PARAM, pass in either: PRES, PSAL, or TEMP as a string
    - Generates PRESSURE v COUNT, PRES v PSAL, or PRES v TEMP graph
  - Ability to flag individual points, click on the same point to cycle through QC flag options (good, probably good, probably bad, bad)
  - Salinity density inversions are marked with edge color "fuchsia"
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
date_filter_start          | Format: "YYYY_MM_DD_HH_MM_SS" or None, ability to filter for a date range for dataset graphs, if date_filter_start is specified and date_filter_end is "None", range will be from start date - end of profile data
date_filter_end            | Format: "YYYY_MM_DD_HH_MM_SS" or None
prof_num_filter            | Format: "PROFNUM_PROFNUM"; EX: "5-7" will get you profiles 5-7. Specify either prof num filter or date filter! 

1. graph_pres_v_var_all
   - Generates a PRES v DATA_TYPE graph for all profiles
   - Ability to click and select profiles of interest
   - Ability to filter by date range and prof num
2. graph_TS_all
   - Generates a TS graph for all profiles
   - Ability to click and select profiles of interest
   - Ability to filter by date range and prof num
3. graph_deep_section_var_all
   - Generates a DEEP SECTION graph for all profiles
   - Ability to filter by date range and prof num

After exiting out of the aforementioned graphs, selected profiles of interest will pop out a data_snapshot_graph, and flag_range_data_graph for TEMP and PSAL where you can manipulate QC flags

## make_KML_files/make_kml.py
Saves a data snapshot image for every profile in the intermediate netCDF directory, then generates a Google Earth KML file with one placemark per profile containing that snapshot as a popup image.

**Output structure:**
```
<save_dir>/
├── images/
│   ├── profile_1_data_snapshot.png
│   ├── profile_2_data_snapshot.png
│   └── ...
└── <float_name>.kml
```

The KML references images via relative paths (`images/profile_N_data_snapshot.png`), so the KML file and the `images/` folder must stay siblings in the same directory.

**Usage:**
```
python make_KML_files/make_kml.py -read_dir <path> -save_dir <path> -float_name <name>
```

FLAG          | DESCRIPTION
------------- | -------------
`-read_dir`   | Directory of intermediate netCDF files to read from
`-save_dir`   | Output directory; images saved to `save_dir/images/`, KML saved to `save_dir/<float_name>.kml`
`-float_name` | Float name used for the KML filename and trajectory label (e.g. `F9186`)

**Example:**
```
python make_KML_files/make_kml.py -read_dir "c:\FLOAT_DATA\F9186\DMODE\F9186_VI" -save_dir "c:\FLOAT_DATA\F9186\DMODE\F9186_KML" -float_name F9186
```

## graphs_nc.py
This module provides all interactive matplotlib-based graphs used during delayed-mode QC. It is called internally by `delayed_mode_processing.py`.
Functions are grouped into three categories:

### Overview Graphs — visualize all profiles at once
These functions display the full dataset and support hover/click to select profiles of interest. Selected profiles are returned as a set and used downstream (e.g. to pop up a `data_snapshot_graph`).

FUNCTION                          | DESCRIPTION
--------------------------------- | -------------
`pres_v_var_all`                  | PRES v TEMP or PSAL for all profiles; click to select profiles
`TS_graph_single_dataset_all_profile` | TS diagram for all profiles of a single float; click to select profiles
`TS_graph_double`                 | TS diagram comparing two floats side-by-side (purple vs red)
`deep_section_var_all`            | Depth section (Hovmuller diagram) of TEMP or PSAL over time

### Interactive QC Flagging — flag points in a single profile
These functions display a single profile and allow the user to assign QC flags by clicking. Clicking a point cycles it through: bad (red) → probably bad (orange) → probably good (aqua) → good (green). Points involved in density inversions are highlighted with a fuchsia edge color.

FUNCTION                    | DESCRIPTION
--------------------------- | -------------
`flag_point_data_graphs`    | Click individual points to cycle their QC flag; supports PRES, TEMP, PSAL; density inversion detection included
`flag_range_data_graphs`    | Click two boundary points to flag a contiguous range; overlapping ranges are auto-merged
`flag_TS_data_graphs`       | Click points on the TS diagram to toggle between both-good, sal-bad, temp-bad, both-bad
`single_prof_datasnapshot`  | 2×2 panel: TEMP flag graph, PSAL flag graph, TS diagram, and profile info text

### Helper Functions

FUNCTION                  | DESCRIPTION
------------------------- | -------------
`density_inversion_test`  | Detects density inversions using TEOS-10 (GSW); returns indices of inverted levels
`merge_ranges`            | Merges overlapping index ranges (used internally by `flag_range_data_graphs`)
`del_bad_points`          | Sets all data levels to NaN where any of PRES, TEMP, or PSAL QC = 4

### QC Color Convention (all interactive graphs)
Color   | QC Value | Meaning
------- | -------- | -------
Green   | 1        | Good
Aqua    | 2        | Probably good
Orange  | 3        | Probably bad
Red     | 4        | Bad
Fuchsia edge | —   | Density inversion detected at this level

## make_final_nc_files.py
This module produces the final delayed-mode ARGO netCDF files from the intermediate files. Configure it via `make_final_nc_config.toml` (see **Configuration Files** above), then run:

```
python make_final_nc_files.py
```

### First-time setup
Set `run_make_config_file = true` once to generate a per-float config file template (or auto-populate it from original real-time ARGO netCDF files if `orgargo_netcdf_filepath` is set). Set it back to `false` for all subsequent runs.

### `orgargo_netcdf_filepath`
If the original real-time ARGO netCDF files are available, set this path so that metadata like `date_creation` is preserved and `SCIENTIFIC_CALIB` / history entries are appended rather than overwritten. Set to `""` to skip.

## tools.py
Shared utilities used across all pipeline scripts. Import with `from tools import ...`.

FUNCTION                      | DESCRIPTION
----------------------------- | -------------
`from_julian_day(julian_day)` | Convert Julian day (1950-01-01 reference) to a timezone-aware UTC datetime; returns NaN if input is NaN
`to_julian_day(date_obj)`     | Convert a datetime object to Julian days since 1950-01-01 UTC
`del_all_nan_slices(argo_data)` | Remove profiles where PRES, TEMP, or PSAL ADJUSTED arrays are entirely NaN; updates all 1D and 2D arrays in the dict
`make_intermediate_nc_file(argo_data, dest_filepath, float_num, profile_num=None)` | Write intermediate netCDF files from an argo_data dict; one file per profile; pass profile_num to write a single profile
`read_intermediate_nc_file(filepath)` | Read all intermediate .nc files in a directory into a single argo_data dict; NaN-pads profiles to equal length

## drift_analysis.py
Salinity drift analysis comparing Argo float salinity against independent reference datasets. This module is designed to be more interactive, workflow optimization is still underway. 

### Reference Data Readers

FUNCTION                   | DESCRIPTION
-------------------------- | -------------
`read_AXCTDs(filepath, bin_size)` | Read one AXCTD .edf drop; optionally bin-average at bin_size dbar intervals
`read_float_11678(txt_file)` | Parse specialized float-11678 data from a text file
`read_regular_file(filepath, date, col_names, bin_size, cndc, lat, lon)` | Read a generic CSV/TXT data file into a standardized dict
`read_corrected_gem_data()` | Read bottle-corrected GEM salinity data from Excel
`read_bottle_data(fp)`     | Read bottle salinity samples from XLS file
`read_lorenze_ctd_data(bin_sizes)` | Read Lorenz CTD data from .cnv files

### Float-Specific Analysis Functions

Each float has two analysis methods and (where applicable) a TS diagram function:

FUNCTION                               | DESCRIPTION
-------------------------------------- | -------------
`generate_F9185_F9444_avg_PSAL()`      | F9185 & F9444: avg PSAL at 500–600 dbar vs AXCTD reference
`generate_F9185_F9444_PSAL_at_TEMP()`  | F9185 & F9444: PSAL at 2°C isotherm vs AXCTD reference
`generate_F10052_avg_PSAL()`           | F10052: avg PSAL at 150–400 dbar vs CTD/GEM/bottle references
`generate_F10052_PSAL_at_TEMP(TT, save_dir)` | F10052: PSAL at isotherm TT vs reference datasets; optional save
`F9444_avg_PSAL()`                     | F9444: avg PSAL at 700–800 dbar vs Nicole CTD and Melville AXCTDs
`F9444_PSAL_AT_TEMP(TT, save_dir)`     | F9444: PSAL at isotherm TT vs reference datasets
`F9444_TS()`                           | F9444: TS diagram vs F9185 and AXCTD reference data
`F9443_avg_PSAL()`                     | F9443: avg PSAL at 500–600 dbar vs ORP WOOD CTD and AXCTDs
`F9443_PSAL_AT_TEMP(TT, save_dir)`     | F9443: PSAL at isotherm TT vs reference datasets
`F9443_TS()`                           | F9443: TS diagram vs float F11678 and ORP WOOD CTD
`generate_F0051_F9186_TS()`            | F10051 & F9186: TS diagram for profiles ≥ 500 dbar, Feb 1 – Jul 31 2025
`generate_F0051_F9186_PSAL_at_TEMP()`  | F10051 & F9186: per-profile least-squares PSAL at 2°C (≥ 500 dbar, TEMP < 2.1°C), plotted vs time

### Helper Utilities

FUNCTION                                                | DESCRIPTION
------------------------------------------------------- | -------------
`dm_to_decimal(deg, minutes, hemisphere)`               | Convert degrees + minutes to decimal degrees
`filter_pres_levels(float1_data, pres_min, pres_max)`   | Set values outside pressure range to NaN across all arrays
`read_float_apply_qc(nc_filepath)`                      | Read intermediate netCDF and apply QC masking (sets QC≥3 to NaN)
`filter_float_overlap_date_range(f1, f1_name, f2, f2_name, just_overlap)` | Filter two float dicts to overlapping or partially overlapping date range
`find_psal_at_temp(target_temp, data, show_graph)`      | Interpolate PSAL at a target isotherm temperature for each profile
`make_TS_plot(list_of_data, list_of_labels)`            | TS diagram with sigma-t density contours for multiple datasets


