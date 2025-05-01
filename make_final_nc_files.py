import copy
import glob
import numpy as np
import netCDF4 as nc4
from datetime import datetime, timezone, timedelta
from tools import from_julian_day, read_intermediate_nc_file, to_julian_day
import os 
import numpy.ma as ma
import re
import gsw 

def masked_byte_array_to_str(masked_array):
    """
    Converts a masked byte array to a regular string.
    
    Parameters:
        masked_array (numpy.ma.MaskedArray or numpy.ndarray): The masked byte array to convert.

    Returns:
        str: The resulting string, with masked elements replaced by an empty string.
    """
    if isinstance(masked_array, ma.MaskedArray):
        # Replace masked values with an empty byte
        filled_array = masked_array.filled(b'')
    else:
        # If it's not masked, use the array as is
        filled_array = masked_array

    # Join the bytes into a single byte string, then decode to UTF-8
    return filled_array.tobytes().decode('utf-8').replace('\x00', ' ').strip()

def make_config_file(float_num, dest_filepath, org_argo_netcdf_filepath = None):
    """
    Makes a config ".txt" file for an associated float, inits unknown vars with "None"
    for user to manually fill in later.

    Args:
        float_num (str): float number 
        dest_filepath (str): destination filepath
        org_argo_netcdf_filepath (str, optional): filepath to an orginal argo netcdf float profile. Defaults to None.
    """

    # Get first file in dir of file 
    if org_argo_netcdf_filepath is not None:
        org_files = glob.glob(os.path.join(org_argo_netcdf_filepath, "*.nc"))
        argo_org_file = nc4.Dataset(org_files[0])

    txt_config_filepath = os.path.join(dest_filepath, f"{float_num}_config_file.txt")
   
    with open(txt_config_filepath, 'w') as txt_file:
        txt_file.write(f"Argo Float {float_num} Configuration File for Delayed Mode File Generation \n")
        txt_file.write("Please make sure all parameters are filled\n")
        txt_file.write("Reference table link: https://archimer.ifremer.fr/doc/00187/29825/ \n")
        txt_file.write(f"Date of generation: {str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}\n")
        txt_file.write("===========================================================================\n")

        if org_argo_netcdf_filepath is None:
            txt_file.write("DATA_CENTRE = None\n")
            txt_file.write("DATA_MODE = None\n")
            txt_file.write("DATA_STATE_INDICATOR = None\n")
            txt_file.write("DATA_TYPE = None\n")
            txt_file.write("DC_REFERENCE = None\n")
            txt_file.write("DIRECTION = None\n")
            txt_file.write("FIRMWARE_VERSION = None\n")
            txt_file.write("FLOAT_SERIAL_NO = None\n")
            txt_file.write("FORMAT_VERSION = None\n")
            txt_file.write("HANDBOOK_VERSION = None\n")
            txt_file.write("PI_NAME = None\n")
            txt_file.write("PLATFORM_NUMBER = None\n")
            txt_file.write("PLATFORM_TYPE = None\n")
            txt_file.write("POSITIONING_SYSTEM = None\n")
            txt_file.write("PROJECT_NAME = None\n")
            txt_file.write("REFERENCE_DATE_TIME = None\n")
            txt_file.write("STATION_PARAMETERS = None\n")
            txt_file.write("PARAMETER = None\n")
            txt_file.write("VERTICAL_SAMPLING_SCHEME = None\n")
            txt_file.write("WMO_INST_TYPE = None\n")
        else: 
            data_centre = argo_org_file.variables['DATA_CENTRE']
            data_centre = np.squeeze(np.char.decode(data_centre[:].filled(data_centre.getncattr("_FillValue"))))
            txt_file.write(f"DATA_CENTRE = {''.join(data_centre)}\n")
            
            data_mode = argo_org_file.variables['DATA_MODE']
            data_mode = np.squeeze(np.char.decode(data_mode[:].filled(data_mode.getncattr("_FillValue"))))
            txt_file.write(f"DATA_MODE = {data_mode}\n")

            data_state_indicator = argo_org_file.variables['DATA_STATE_INDICATOR']
            datadata_state_indicator = np.squeeze(np.char.decode(data_state_indicator[:].filled(data_state_indicator.getncattr("_FillValue"))))
            txt_file.write(f"DATA_STATE_INDICATOR = {''.join(datadata_state_indicator)}\n")

            data_type = argo_org_file.variables['DATA_TYPE']
            data_type = np.squeeze(np.char.decode(data_type[:].filled(data_type.getncattr("_FillValue"))))
            txt_file.write(f"DATA_TYPE = {''.join(data_type)}\n")

            dc_reference = argo_org_file.variables['DC_REFERENCE']
            dc_reference = np.squeeze(np.char.decode(dc_reference[:].filled(dc_reference.getncattr("_FillValue"))))
            txt_file.write(f"DC_REFERENCE = {''.join(dc_reference)}\n")

            direction = argo_org_file.variables['DIRECTION']
            direction = np.squeeze(np.char.decode(direction[:].filled(direction.getncattr("_FillValue"))))
            txt_file.write(f"DIRECTION = {direction}\n")

            firmware_version = argo_org_file.variables['FIRMWARE_VERSION'] 
            firmware_version = np.squeeze(np.char.decode(firmware_version[:].filled(firmware_version.getncattr("_FillValue"))))
            txt_file.write(f"FIRMWARE_VERSION = {''.join(firmware_version)}\n")

            float_serial_no = argo_org_file.variables['FLOAT_SERIAL_NO']
            float_serial_no = np.squeeze(np.char.decode(float_serial_no[:].filled(float_serial_no.getncattr("_FillValue"))))
            txt_file.write(f"FLOAT_SERIAL_NO = {''.join(float_serial_no)}\n")

            format_version = argo_org_file.variables['FORMAT_VERSION']
            format_version = np.squeeze(np.char.decode(format_version[:].filled(format_version.getncattr("_FillValue"))))
            txt_file.write(f"FORMAT_VERSION = {''.join(format_version)}\n")

            handbook_version = argo_org_file.variables['HANDBOOK_VERSION']
            handbook_version = np.squeeze(np.char.decode(handbook_version[:].filled(handbook_version.getncattr("_FillValue"))))
            txt_file.write(f"HANDBOOK_VERSION = {''.join(handbook_version)}\n")

            pi_name = argo_org_file.variables['PI_NAME']
            pi_name = np.squeeze(np.char.decode(pi_name[:].filled(pi_name.getncattr("_FillValue"))))
            txt_file.write(f"PI_NAME = {''.join(pi_name)}\n")

            platform_number = argo_org_file.variables['PLATFORM_NUMBER']
            platform_number = np.squeeze(np.char.decode(platform_number[:].filled(platform_number.getncattr("_FillValue"))))
            txt_file.write(f"PLATFORM_NUMBER = {''.join(platform_number)}\n")

            platform_type = argo_org_file.variables['PLATFORM_TYPE']
            platform_type = np.squeeze(np.char.decode(platform_type[:].filled(platform_type.getncattr("_FillValue"))))
            txt_file.write(f"PLATFORM_TYPE = {''.join(platform_type)}\n")

            positioning_system = argo_org_file.variables['POSITIONING_SYSTEM']
            positioning_system = np.squeeze(np.char.decode(positioning_system[:].filled(positioning_system.getncattr("_FillValue"))))
            txt_file.write(f"POSITIONING_SYSTEM = {''.join(positioning_system)}\n")

            project_name = argo_org_file.variables['PROJECT_NAME']
            project_name = np.squeeze(np.char.decode(project_name[:].filled(project_name.getncattr("_FillValue"))))
            txt_file.write(f"PROJECT_NAME = {''.join(project_name)}\n")

            reference_date_time = argo_org_file.variables['REFERENCE_DATE_TIME']
            reference_date_time = np.squeeze(np.char.decode(reference_date_time[:].filled(reference_date_time.getncattr("_FillValue"))))
            txt_file.write(f"REFERENCE_DATE_TIME = {''.join(reference_date_time)}\n")

            station_parameters = argo_org_file.variables['STATION_PARAMETERS']
            station_parameters = np.squeeze(np.char.decode(station_parameters[:].filled(station_parameters.getncattr("_FillValue"))))
            temp = ''
            for a in station_parameters:
                word = re.sub(r"\s+", " ", ''.join(a))
                temp = temp + word
            txt_file.write(f"STATION_PARAMETERS = {temp.replace(" ", ", ")[:-2]}\n")

            parameter = argo_org_file.variables['PARAMETER']
            parameter = np.squeeze(np.char.decode(parameter[:].filled(parameter.getncattr("_FillValue"))))
            temp = ''
            for a in parameter:    
                for b in a:
                    word = re.sub(r"\s+", " ", ''.join(b))
                    temp = temp + word
            txt_file.write(f"PARAMETER = {re.sub(r"\s{1,}", ", ", temp)[:-2]}\n")

            vertical_sampling_scheme = argo_org_file.variables['VERTICAL_SAMPLING_SCHEME']
            vertical_sampling_scheme = np.squeeze(np.char.decode(vertical_sampling_scheme[:].filled(vertical_sampling_scheme.getncattr("_FillValue"))))
            txt_file.write(f"VERTICAL_SAMPLING_SCHEME = {''.join(vertical_sampling_scheme)}\n")

            wmo_inst_type = argo_org_file.variables['WMO_INST_TYPE']
            wmo_inst_type = np.squeeze(np.char.decode(wmo_inst_type[:].filled(wmo_inst_type.getncattr("_FillValue"))))
            txt_file.write(f"WMO_INST_TYPE = {''.join(wmo_inst_type)}\n")
        
        txt_file.write(f"===========================ADJUSTED_ERROR_PARAMS===========================\n")
        txt_file.write(f"CNDC_ADJUSTED_ERROR = None\n")
        txt_file.write(f"PRES_ADJUSTED_ERROR = None\n")
        txt_file.write(f"PSAL_ADJUSTED_ERROR = None\n")
        txt_file.write(f"TEMP_ADJUSTED_ERROR = None\n")

def history_qc_test_converter(input_value, mode='decode'):
    """
    Maps between a list of QC tests and their associated numerical values. Takes said list
    and converts them to a hex number (added value of QC test's associated numerical value,
    converted to a hex number). This mapping goes both ways, so passing in a hex number and setting
    mode = 'decode' returns list of QC tests performed.

    Args:
        mode (str, optional): decode, default val
            - input_val (str): hex number
            - returns a list of QC tests performed
        mode (str, optional): encode
            - input_val (list, str): list of qc_tests names
            - returns str hex number

    Raises:
        ValueError: raised when mode passed in is not 'decode' or 'encode'
    """
    qc_tests = {
        2: "Platform Identification test",
        4: "Impossible Date test",
        8: "Impossible Location test",
        16: "Position on Land test",
        32: "Impossible Speed test",
        64: "Global Range test",
        128: "Regional Global Parameter test",
        256: "Pressure Increasing test",
        512: "Spike test",
        1024: "Top and Bottom Spike test (obsolete)",
        2048: "Gradient test",
        4096: "Digit Rollover test",
        8192: "Stuck Value test",
        16384: "Density Inversion test",
        32768: "Grey List test",
        65536: "Gross Salinity or Temperature Sensor Drift test",
        131072: "Visual QC test",
        261144: "Frozen profile test",
        524288: "Deepest pressure test",
        1048576: "Questionable Argos position test",
        2097152: "Near-surface unpumped CTD salinity test",
        4194304: "Near-surface mixed air/water test"
    }

    if mode == 'decode':
        decimal_value = int(input_value, 16)
        active_tests = []
        for qc_id, name in sorted(qc_tests.items(), reverse=True):
            if decimal_value >= qc_id:
                active_tests.append(name)
                decimal_value -= qc_id  # Subtract the matched QC ID
        return active_tests
    
    elif mode == 'encode':
        binary_num = 0
        for qc_id, name in qc_tests.items():
            if name in input_value:
                binary_num = binary_num + qc_id
        hex_num = hex(binary_num)   
        return hex_num[2:]
    
    else:
        raise ValueError("Invalid mode. Use 'decode' or 'encode'.")

def make_final_nc_files(final_nc_data_prof, float_num, dest_filepath):
    """
    Makes the final version of ARGO Delayed Mode Processed NETCDF File. Variables have
    complete description names and such.

    Args:
        final_nc_data_prof (dict): dictionary of all associated parameters needed to generate file.
        float_num (str): float number
        dest_filepath (str): destination filepath
    """


    output_filename = os.path.join(dest_filepath, f"D{float_num}_{final_nc_data_prof["CYCLE_NUMBER"]:03}.nc")
    nc = nc4.Dataset(output_filename, 'w', format="NETCDF4")

    # Create dimensions
    nc.createDimension('DATE_TIME', 14)
    nc.createDimension('STRING256', 256)
    nc.createDimension('STRING64', 64)
    nc.createDimension('STRING32', 32)
    nc.createDimension('STRING16', 16)
    nc.createDimension('STRING8', 8)
    nc.createDimension('STRING4', 4)
    nc.createDimension('STRING2', 2)
    nc.createDimension('N_PROF', 1)
    nc.createDimension('N_PARAM', len(final_nc_data_prof["PARAMETER"].split(', ')))
    nc.createDimension('N_LEVELS', len(final_nc_data_prof["PRES"]))
    nc.createDimension('N_HISTORY', None)
    nc.createDimension('N_CALIB', final_nc_data_prof["SCIENTIFIC_CALIB_COEFFICIENT"].shape[0])

    CNDC = nc.createVariable('CNDC', 'f4', ('N_PROF', 'N_LEVELS'), fill_value=99999.0)
    CNDC.long_name = "Electrical conductivity"
    CNDC.standard_name = "sea_water_electrical_conductivity"
    CNDC.units = "mhos/m"
    CNDC.valid_min = np.float32(0.0)
    CNDC.valid_max = np.float32(8.5)
    CNDC.C_format = "%12.5f"
    CNDC.FORTRAN_format = "F12.5"
    CNDC.resolution = np.float32(1.0e-4)
    CNDC[:] = final_nc_data_prof["CNDC"]

    CNDC_ADJUSTED = nc.createVariable('CNDC_ADJUSTED', 'f4', ('N_PROF', 'N_LEVELS'), fill_value=99999.0)
    CNDC_ADJUSTED.long_name = "Electrical conductivity"
    CNDC_ADJUSTED.standard_name = "sea_water_electrical_conductivity"
    CNDC_ADJUSTED.units = "mhos/m"
    CNDC_ADJUSTED.valid_min = np.float32(0.0)
    CNDC_ADJUSTED.valid_max = np.float32(8.5)
    CNDC_ADJUSTED.C_format = "%12.5f"
    CNDC_ADJUSTED.FORTRAN_format = "F12.5"
    CNDC_ADJUSTED.resolution = np.float32(1.0e-4)
    CNDC_ADJUSTED[:] = final_nc_data_prof["CNDC_ADJUSTED"]

    CNDC_ADJUSTED_ERROR = nc.createVariable('CNDC_ADJUSTED_ERROR', 'f4', ('N_PROF', 'N_LEVELS'), fill_value=99999.0)
    CNDC_ADJUSTED_ERROR.long_name = "Contains the error on the adjusted values as determined by the delayed mode QC process"
    CNDC_ADJUSTED_ERROR.units = "mhos/m"
    CNDC_ADJUSTED_ERROR.C_format = "%12.5f"
    CNDC_ADJUSTED_ERROR.FORTRAN_format = "F12.5"
    CNDC_ADJUSTED_ERROR.resolution = np.float32(1.0e-4)
    CNDC_ADJUSTED_ERROR[:] = final_nc_data_prof["CNDC_ADJUSTED_ERROR"]

    CNDC_ADJUSTED_QC = nc.createVariable('CNDC_ADJUSTED_QC', 'S1', ('N_PROF', 'N_LEVELS'), fill_value = " ")
    CNDC_ADJUSTED_QC.long_name = "quality flag"
    CNDC_ADJUSTED_QC.conventions = "Argo reference table 2"
    CNDC_ADJUSTED_QC[:] = final_nc_data_prof["CNDC_ADJUSTED_QC"].astype('S1')

    CNDC_QC = nc.createVariable('CNDC_QC', 'S1', ('N_PROF', 'N_LEVELS'), fill_value = " ")
    CNDC_QC.long_name = "quality flag"
    CNDC_QC.conventions = "Argo reference table 2"
    CNDC_QC[:] = final_nc_data_prof["CNDC_QC"].astype('S1')

    CONFIG_MISSION_NUMBER = nc.createVariable('CONFIG_MISSION_NUMBER', 'i4', ('N_PROF'), fill_value=99999)
    CONFIG_MISSION_NUMBER.long_name = "Unique number denoting the missions performed by the float"
    CONFIG_MISSION_NUMBER.conventions = "1...N, 1 : first complete mission"
    CONFIG_MISSION_NUMBER[:] = final_nc_data_prof["CONFIG_MISSION_NUMBER"]
            
    CYCLE_NUMBER = nc.createVariable('CYCLE_NUMBER', 'i4', ('N_PROF'), fill_value=99999)
    CYCLE_NUMBER.long_name = "Float cycle number"
    CYCLE_NUMBER.conventions = "0...N, 0 : launch cycle (if exists), 1 : first complete cycle"
    CYCLE_NUMBER[:] = final_nc_data_prof["CYCLE_NUMBER"]

    DATA_CENTRE = nc.createVariable('DATA_CENTRE', 'S1', ('N_PROF', 'STRING2'), fill_value = " ")
    DATA_CENTRE.long_name = "Data centre in charge of float data processing"
    DATA_CENTRE.conventions = "Argo reference table 4"
    DATA_CENTRE[:] = np.array(list(final_nc_data_prof["DATA_CENTRE"]), dtype='S1')

    DATA_MODE = nc.createVariable('DATA_MODE', 'S1', ('N_PROF'), fill_value=" ")
    DATA_MODE.long_name = "Delayed mode or real time data"
    DATA_MODE.conventions = "R : real time; D : delayed mode; A : real time with adjustment"
    DATA_MODE[:] = 'D'

    DATA_STATE_INDICATOR = nc.createVariable('DATA_STATE_INDICATOR', 'S1', ('N_PROF', 'STRING4'), fill_value=" ")
    DATA_STATE_INDICATOR.long_name = "Degree of processing the data have passed through"
    DATA_STATE_INDICATOR.conventions = "Argo reference table 6"
    DATA_STATE_INDICATOR[:] = list('  2B')

    DATA_TYPE = nc.createVariable('DATA_TYPE', 'S1', ('STRING16'), fill_value=" ")
    DATA_TYPE.long_name = "Data type"
    DATA_TYPE.conventions = "Argo reference table 1"
    DATA_TYPE[:] = list('Argo profile    ')

    DATE_CREATION = nc.createVariable('DATE_CREATION', 'S1', ('DATE_TIME'), fill_value=" ")
    DATE_CREATION.long_name = "Date of file creation"
    DATE_CREATION.conventions = "YYYYMMDDHHMISS"
    DATE_CREATION[:] = final_nc_data_prof["DATE_CREATION"]
  
    DATE_UPDATE = nc.createVariable('DATE_UPDATE', 'S1', ('DATE_TIME'), fill_value=" ")
    DATE_UPDATE.long_name = "Date of update of this file"
    DATE_UPDATE.conventions = "YYYYMMDDHHMISS"
    DATE_UPDATE[:] = final_nc_data_prof["DATE_UPDATE"]

    DC_REFERENCE = nc.createVariable('DC_REFERENCE', 'S1', ('N_PROF', 'STRING32'), fill_value=" ")
    DC_REFERENCE.long_name = "Station unique identifier in data centre"
    DC_REFERENCE.conventions = "Data centre convention"
    DC_REFERENCE[:] = np.array(np.pad(list(final_nc_data_prof["DC_REFERENCE"]), (0, 32 - len(final_nc_data_prof["DC_REFERENCE"])), mode='constant', constant_values=' '), dtype='S1')

    DIRECTION = nc.createVariable('DIRECTION', 'S1', ('N_PROF'), fill_value=" ")
    DIRECTION.long_name = "Direction of the station profiles"
    DIRECTION.conventions = "A: ascending profiles, D: descending profiles"
    DIRECTION[:] = 'A' 

    FIRMWARE_VERSION = nc.createVariable('FIRMWARE_VERSION', 'S1', ('N_PROF', 'STRING32'), fill_value=" ")
    FIRMWARE_VERSION.long_name = "Instrument firmware version"
    FIRMWARE_VERSION[:] = np.array(np.pad(list(final_nc_data_prof["FIRMWARE_VERSION"]), (0, 32 - len(final_nc_data_prof["FIRMWARE_VERSION"])), mode='constant', constant_values=' '), dtype='S1')

    FLOAT_SERIAL_NO = nc.createVariable('FLOAT_SERIAL_NO', 'S1', ('N_PROF', 'STRING32'), fill_value=" ")
    FLOAT_SERIAL_NO.long_name = "Serial number of the float"
    FLOAT_SERIAL_NO[:] = np.array(np.pad(list(final_nc_data_prof["FLOAT_SERIAL_NO"]), (0, 32 - len(final_nc_data_prof["FLOAT_SERIAL_NO"])), mode='constant', constant_values=' '), dtype='S1')

    FORMAT_VERSION = nc.createVariable('FORMAT_VERSION', 'S1', ('STRING4'), fill_value=" ")
    FORMAT_VERSION.long_name = "File format version"
    FORMAT_VERSION[:] = np.array(np.pad(list(final_nc_data_prof["FORMAT_VERSION"]), (0, 4 - len(final_nc_data_prof["FORMAT_VERSION"])), mode='constant', constant_values=' '), dtype='S1')

    HANDBOOK_VERSION = nc.createVariable('HANDBOOK_VERSION', 'S1', ('STRING4'), fill_value=" ")
    HANDBOOK_VERSION.long_name = "Data handbook version"
    HANDBOOK_VERSION[:] = np.array(np.pad(list(final_nc_data_prof["HANDBOOK_VERSION"]), (0, 4 - len(final_nc_data_prof["HANDBOOK_VERSION"])), mode='constant', constant_values=' '), dtype='S1')

    HISTORY_ACTION = nc.createVariable('HISTORY_ACTION', 'S1', ('N_HISTORY', 'N_PROF', 'STRING4'), fill_value=" ")
    HISTORY_ACTION.long_name = "Action performed on data"
    HISTORY_ACTION.conventions = "Argo reference table 7"
    HISTORY_ACTION[:] = np.array([list(element) for element in final_nc_data_prof["HISTORY_ACTION"]], dtype='S1')

    HISTORY_DATE = nc.createVariable('HISTORY_DATE', 'S1', ('N_HISTORY', 'N_PROF', 'DATE_TIME'), fill_value=" ")
    HISTORY_DATE.long_name = "Date the history record was created"
    HISTORY_DATE.conventions = "YYYYMMDDHHMISS"
    HISTORY_DATE[:] = np.array([list(element) for element in final_nc_data_prof["HISTORY_DATE"]], dtype='S1')

    HISTORY_INSTITUTION = nc.createVariable('HISTORY_INSTITUTION', 'S1', ('N_HISTORY', 'N_PROF', 'STRING4'), fill_value=" ")
    HISTORY_INSTITUTION.long_name = "Institution which performed action"
    HISTORY_INSTITUTION.conventions = "Argo reference table 4"
    HISTORY_INSTITUTION[:] = np.array([list(element) for element in final_nc_data_prof["HISTORY_INSTITUTION"]], dtype='S1')

    HISTORY_PARAMETER = nc.createVariable('HISTORY_PARAMETER', 'S1', ('N_HISTORY', 'N_PROF', 'STRING16'), fill_value=" ")
    HISTORY_PARAMETER.long_name = "Station parameter action is performed on"
    HISTORY_PARAMETER.conventions = "Argo reference table 3"
    HISTORY_PARAMETER[:] = np.array([list(element) for element in final_nc_data_prof["HISTORY_PARAMETER"]], dtype='S1')

    HISTORY_PREVIOUS_VALUE = nc.createVariable('HISTORY_PREVIOUS_VALUE', 'f4', ('N_HISTORY', 'N_PROF'), fill_value = 99999.0)
    HISTORY_PREVIOUS_VALUE.long_name = "Parameter/Flag previous value before action"
    HISTORY_PREVIOUS_VALUE[:] = np.array(final_nc_data_prof["HISTORY_PREVIOUS_VALUE"])

    HISTORY_QCTEST = nc.createVariable('HISTORY_QCTEST', 'S1', ('N_HISTORY', 'N_PROF', 'STRING16'), fill_value=" ")
    HISTORY_QCTEST.long_name = "Documentation of tests performed, tests failed (in hex form)"
    HISTORY_QCTEST.conventions = "Write tests performed when ACTION=QCP$; tests failed when ACTION=QCF$"
    HISTORY_QCTEST[:] = np.array([list(element) for element in final_nc_data_prof["HISTORY_QCTEST"]], dtype='S1')

    HISTORY_REFERENCE = nc.createVariable('HISTORY_REFERENCE', 'S1', ('N_HISTORY', 'N_PROF', 'STRING64'), fill_value=" ")
    HISTORY_REFERENCE.long_name = "Reference of database"
    HISTORY_REFERENCE.conventions = "Institution dependent"
    HISTORY_REFERENCE[:] = np.array([list(element) for element in final_nc_data_prof["HISTORY_REFERENCE"]], dtype='S1')

    HISTORY_SOFTWARE = nc.createVariable('HISTORY_SOFTWARE', 'S1', ('N_HISTORY', 'N_PROF', 'STRING4'), fill_value=" ")
    HISTORY_SOFTWARE.long_name = "Name of software which performed action"
    HISTORY_SOFTWARE.conventions = "Institution dependent"
    HISTORY_SOFTWARE[:] = np.array([list(element) for element in final_nc_data_prof["HISTORY_SOFTWARE"]], dtype='S1')

    HISTORY_SOFTWARE_RELEASE = nc.createVariable('HISTORY_SOFTWARE_RELEASE', 'S1', ('N_HISTORY', 'N_PROF', 'STRING4'), fill_value=" ")
    HISTORY_SOFTWARE_RELEASE.long_name = "Version/release of software which performed action"
    HISTORY_SOFTWARE_RELEASE.conventions = "Institution dependent"
    HISTORY_SOFTWARE_RELEASE[:] = np.array([list(element) for element in final_nc_data_prof["HISTORY_SOFTWARE_RELEASE"]], dtype='S1')

    HISTORY_START_PRES = nc.createVariable('HISTORY_START_PRES', 'f4', ('N_HISTORY', 'N_PROF'), fill_value=99999.0)
    HISTORY_START_PRES.long_name = "Start pressure action applied on"
    HISTORY_START_PRES.units = "decibar"
    HISTORY_START_PRES[:] = final_nc_data_prof["HISTORY_START_PRES"].astype('f4')

    HISTORY_STEP = nc.createVariable('HISTORY_STEP', 'S1', ('N_HISTORY', 'N_PROF', 'STRING4'), fill_value=" ")
    HISTORY_STEP.long_name = "Step in data processing"
    HISTORY_STEP.conventions = "Argo reference table 12"
    HISTORY_STEP[:] = np.array([list(element) for element in final_nc_data_prof["HISTORY_STEP"]], dtype='S1')

    HISTORY_STOP_PRES = nc.createVariable('HISTORY_STOP_PRES', 'f4', ('N_HISTORY', 'N_PROF'), fill_value=99999.0)
    HISTORY_STOP_PRES.long_name = "Stop pressure action applied on"
    HISTORY_STOP_PRES.units = "decibar"
    HISTORY_STOP_PRES[:] = final_nc_data_prof["HISTORY_STOP_PRES"].astype('f4')

    JULD = nc.createVariable('JULD', 'f8', ('N_PROF'), fill_value=99999.0)
    JULD.long_name = "Julian day (UTC) of the station relative to REFERENCE_DATE_TIME"
    JULD.standard_name = "time"
    JULD.units = "days since 1950-01-01 00:00:00 UTC"
    JULD.conventions = "Relative julian days with decimal part (as parts of day)"
    JULD.resolution = 1.1574074074074073E-5
    JULD.axis = "T" 
    JULD[:] = final_nc_data_prof["JULD"]

    JULD_LOCATION = nc.createVariable('JULD_LOCATION', 'f8', ('N_PROF'), fill_value=999999.0)
    JULD_LOCATION.long_name = "Julian day (UTC) of the location relative to REFERENCE_DATE_TIME"
    JULD_LOCATION.units = "days since 1950-01-01 00:00:00 UTC"
    JULD_LOCATION.conventions = "Relative julian days with decimal part (as parts of day)"
    JULD_LOCATION.resolution = 1.1574074074074073E-5
    JULD_LOCATION[:] = final_nc_data_prof["JULD_LOCATION"]

    JULD_QC = nc.createVariable('JULD_QC', 'S1', ('N_PROF'), fill_value=" ")
    JULD_QC.long_name = "Quality on date and time"
    JULD_QC.conventions = "Argo reference table 2"
    JULD_QC[:] = final_nc_data_prof["JULD_QC"]

    LATITUDE = nc.createVariable('LATITUDE', 'f8', ('N_PROF'), fill_value=99999.0)
    LATITUDE.long_name = "Latitude of the station, best estimate"
    LATITUDE.standard_name = "latitude"
    LATITUDE.units = "degree_north"
    LATITUDE.valid_min = -90.0
    LATITUDE.valid_max = 90.0
    LATITUDE.axis = "Y"
    LATITUDE[:] = final_nc_data_prof["LATITUDE"]

    LONGITUDE = nc.createVariable('LONGITUDE', 'f8', ('N_PROF'), fill_value=99999.0)
    LONGITUDE.long_name = "Longitude of the station, best estimate"
    LONGITUDE.standard_name = "longitude"
    LONGITUDE.units = "degree_east"
    LONGITUDE.valid_min = -180.0
    LONGITUDE.valid_max = 180.0
    LONGITUDE.axis = "X"
    LONGITUDE[:] = final_nc_data_prof["LONGITUDE"]

    NB_SAMPLE_CTD = nc.createVariable('NB_SAMPLE_CTD', 'i2', ('N_PROF', 'N_LEVELS'), fill_value=-32767)
    NB_SAMPLE_CTD.long_name = "Number of samples in each pressure bin for the CTD"
    NB_SAMPLE_CTD.units = "count"
    NB_SAMPLE_CTD.C_format = "%5d"
    NB_SAMPLE_CTD.FORTRAN_format = "I5"
    NB_SAMPLE_CTD.resolution = np.int16(1)
    NB_SAMPLE_CTD[:] = final_nc_data_prof["NB_SAMPLE_CTD"].astype('i2')

    NB_SAMPLE_CTD_QC = nc.createVariable('NB_SAMPLE_CTD_QC', 'S1', ('N_PROF', 'N_LEVELS'), fill_value=" ")
    NB_SAMPLE_CTD_QC.long_name = "quality flag"
    NB_SAMPLE_CTD_QC.convetions = "Argo reference table 2"
    NB_SAMPLE_CTD_QC[:] = final_nc_data_prof["NB_SAMPLE_CTD_QC"].astype('S1')

    PARAMETER = nc.createVariable('PARAMETER', 'S1', ('N_PROF', 'N_CALIB', 'N_PARAM', 'STRING16'), fill_value=" ")
    PARAMETER.long_name = "List of parameters with calibration information"
    PARAMETER.conventions = "Argo reference table 3"
    PARAMETER_temp = []
    for param in final_nc_data_prof["PARAMETER"].split(', '):
        param_pad = np.pad(list(param), (0, 16 - len(param)), mode='constant', constant_values=' ')
        PARAMETER_temp.append(param_pad)
    PARAMETER_temp = np.array(PARAMETER_temp, dtype='S1')   
    PARAMETER_val = np.full((final_nc_data_prof["SCIENTIFIC_CALIB_COEFFICIENT"].shape[0], len(final_nc_data_prof["PARAMETER"].split(', ')), 16), ' ', dtype='S1')
    PARAMETER_val[0, :, :] = PARAMETER_temp
    PARAMETER[:] = PARAMETER_val

    PI_NAME = nc.createVariable('PI_NAME', 'S1', ('N_PROF', 'STRING64'), fill_value=" ")
    PI_NAME.long_name = "Name of the principal investigator"
    PI_NAME[:] = np.array(np.pad(list(final_nc_data_prof["PI_NAME"]), (0, 64 - len(final_nc_data_prof["PI_NAME"])), mode='constant', constant_values=' '), dtype='S1')

    PLATFORM_NUMBER = nc.createVariable('PLATFORM_NUMBER', 'S1', ('N_PROF', 'STRING8'), fill_value=" ")
    PLATFORM_NUMBER.long_name = "Float unique identifier"
    PLATFORM_NUMBER.conventions = "WMO float identifier : A9IIIII"
    PLATFORM_NUMBER[:] = np.array(np.pad(list(final_nc_data_prof["PLATFORM_NUMBER"]), (0, 8 - len(final_nc_data_prof["PLATFORM_NUMBER"])), mode='constant', constant_values=' '), dtype='S1')

    PLATFORM_TYPE = nc.createVariable('PLATFORM_TYPE', 'S1', ('N_PROF', 'STRING32'), fill_value=" ")
    PLATFORM_TYPE.long_name = "Type of float"
    PLATFORM_TYPE.conventions = "Argo reference table 23"
    PLATFORM_TYPE[:] = np.array(np.pad(list(final_nc_data_prof["PLATFORM_TYPE"]), (0, 32 - len(final_nc_data_prof["PLATFORM_TYPE"])), mode='constant', constant_values=' '), dtype='S1')

    POSITION_QC = nc.createVariable('POSITION_QC', 'S1', ('N_PROF'), fill_value=" ")
    POSITION_QC.long_name = "Quality on position (latitude and longitude)"
    POSITION_QC.conventions = "Argo reference table 2"
    POSITION_QC[:] = final_nc_data_prof["POSITION_QC"]

    POSITIONING_SYSTEM = nc.createVariable('POSITIONING_SYSTEM', 'S1', ('N_PROF', 'STRING8'), fill_value=" ")
    POSITIONING_SYSTEM.long_name = "Positioning system"
    POSITIONING_SYSTEM[:] = np.array(np.pad(list(final_nc_data_prof["POSITIONING_SYSTEM"]), (0, 8 - len(final_nc_data_prof["POSITIONING_SYSTEM"])), mode='constant', constant_values=' '), dtype='S1')

    PRES = nc.createVariable('PRES', 'f4', ('N_PROF', 'N_LEVELS'), fill_value=99999.0)
    PRES.long_name = "Sea water pressure, equals 0 at sea-level"
    PRES.standard_name = "sea_water_pressure"
    PRES.units = "decibar"
    PRES.valid_min = np.float32(0.0)
    PRES.valid_max = np.float32(12000.0)
    PRES.C_format = "%7.1f"
    PRES.FORTRAN_format = "F7.1"
    PRES.resolution = np.float32(-0.001001001)
    PRES.axis = "Z"
    PRES[:] = final_nc_data_prof["PRES"]

    PRES_ADJUSTED = nc.createVariable('PRES_ADJUSTED', 'f4', ('N_PROF', 'N_LEVELS'), fill_value=99999.0)
    PRES_ADJUSTED.long_name = "Contains the error on the adjusted values as determined by the delayed mode QC process"
    PRES_ADJUSTED.units = "decibar"
    PRES_ADJUSTED.C_format = "%7.1f"
    PRES_ADJUSTED.FORTRAN_format = "F7.1"
    PRES_ADJUSTED.resolution = np.float32(-0.001001001)
    PRES_ADJUSTED[:] = final_nc_data_prof["PRES_ADJUSTED"]

    PRES_ADJUSTED_ERROR = nc.createVariable('PRES_ADJUSTED_ERROR', 'f4', ('N_PROF', 'N_LEVELS'), fill_value=99999.0)
    PRES_ADJUSTED_ERROR.long_name = "Contains the error on the adjusted values as determined by the delayed mode QC process"
    PRES_ADJUSTED_ERROR.units = "decibar"
    PRES_ADJUSTED_ERROR.C_format = "%7.1f"
    PRES_ADJUSTED_ERROR.FORTRAN_format = "F7.1"
    PRES_ADJUSTED_ERROR.resolution = np.float32(-0.001001001)
    PRES_ADJUSTED_ERROR[:] = final_nc_data_prof["PRES_ADJUSTED_ERROR"]

    PRES_ADJUSTED_QC = nc.createVariable('PRES_ADJUSTED_QC', 'S1', ('N_PROF', 'N_LEVELS'), fill_value=" ")
    PRES_ADJUSTED_QC.long_name = "quality flag"
    PRES_ADJUSTED_QC.conventions = "Argo reference table 2"
    PRES_ADJUSTED_QC[:] = final_nc_data_prof["PRES_ADJUSTED_QC"].astype('S1')

    PRES_QC = nc.createVariable('PRES_QC', 'S1', ('N_PROF', 'N_LEVELS'), fill_value=" ")
    PRES_QC.long_name = "quality flag"
    PRES_QC.conventions = "Argo reference table 2"
    PRES_QC[:] = final_nc_data_prof["PRES_QC"].astype('S1')

    PROFILE_CNDC_QC = nc.createVariable('PROFILE_CNDC_QC', 'S1', ('N_PROF'), fill_value=" ")
    PROFILE_CNDC_QC.long_name = "Global quality flag of CNDC profile"
    PROFILE_CNDC_QC.conventions = "Argo reference table 2a"
    PROFILE_CNDC_QC[:] = final_nc_data_prof["PROFILE_CNDC_QC"]

    PROFILE_NB_SAMPLE_CTD_QC = nc.createVariable('PROFILE_NB_SAMPLE_CTD_QC', 'S1', ('N_PROF'), fill_value=" ")
    PROFILE_NB_SAMPLE_CTD_QC.long_name = "Global quality flag of PROFILE_NB_SAMPLE_CTD_QC profile"
    PROFILE_NB_SAMPLE_CTD_QC.conventions = "Argo reference table 2a"
    PROFILE_NB_SAMPLE_CTD_QC[:] = final_nc_data_prof["PROFILE_NB_SAMPLE_CTD_QC"]

    PROFILE_PRES_QC = nc.createVariable('PROFILE_PRES_QC', 'S1', ('N_PROF'), fill_value=" ")
    PROFILE_PRES_QC.long_name = "Global quality flag of PRES profile"
    PROFILE_PRES_QC.conventions = "Argo reference table 2a"
    PROFILE_PRES_QC[:] = final_nc_data_prof["PROFILE_PRES_QC"]

    PROFILE_PSAL_QC = nc.createVariable('PROFILE_PSAL_QC', 'S1', ('N_PROF'), fill_value=" ")
    PROFILE_PSAL_QC.long_name = "Global quality flag of PSAL profile"
    PROFILE_PSAL_QC.conventions = "Argo reference table 2a"
    PROFILE_PSAL_QC[:] = final_nc_data_prof["PROFILE_PSAL_QC"]

    PROFILE_TEMP_CNDC_QC = nc.createVariable('PROFILE_TEMP_CNDC_QC', 'S1', ('N_PROF'), fill_value=" ")
    PROFILE_TEMP_CNDC_QC.long_name = "Global quality flag of TEMP_CNDC profile"
    PROFILE_TEMP_CNDC_QC.conventions = "Argo reference table 2a"
    PROFILE_TEMP_CNDC_QC[:] = final_nc_data_prof["PROFILE_TEMP_CNDC_QC"]

    PROFILE_TEMP_QC = nc.createVariable('PROFILE_TEMP_QC', 'S1', ('N_PROF'), fill_value=" ")
    PROFILE_TEMP_QC.long_name = "Global quality flag of TEMP profile"
    PROFILE_TEMP_QC.conventions = "Argo reference table 2a"
    PROFILE_TEMP_QC[:] = final_nc_data_prof["PROFILE_TEMP_QC"]

    PROJECT_NAME = nc.createVariable('PROJECT_NAME', 'S1', ('N_PROF', 'STRING64'), fill_value=" ")
    PROJECT_NAME.long_name = "Name of the project"
    PROJECT_NAME[:] = np.array(np.pad(list(final_nc_data_prof["PROJECT_NAME"]), (0, 64 - len(final_nc_data_prof["PROJECT_NAME"])), mode='constant', constant_values=' '), dtype='S1')

    PSAL = nc.createVariable('PSAL', 'f4', ('N_PROF', 'N_LEVELS'), fill_value=99999.0)
    PSAL.long_name = "Practical salinity"
    PSAL.standard_name = "sea_water_salinity"
    PSAL.units = "psu"
    PSAL.valid_min = np.float32(2.0)
    PSAL.valid_max = np.float32(41.0)
    PSAL.C_format = "%10.3f"
    PSAL.FORTRAN_format = "F10.3"
    PSAL.resolution = np.float32(-0.001001001)
    PSAL[:] = final_nc_data_prof["PSAL"]

    PSAL_ADJUSTED = nc.createVariable('PSAL_ADJUSTED', 'f4', ('N_PROF', 'N_LEVELS'), fill_value=99999.0)
    PSAL_ADJUSTED.long_name = "Practical salinity"
    PSAL_ADJUSTED.standard_name = "sea_water_salinity"
    PSAL_ADJUSTED.units = "psu"
    PSAL_ADJUSTED.valid_min = np.float32(2.0)
    PSAL_ADJUSTED.valid_max = np.float32(41.0)
    PSAL_ADJUSTED.C_format = "%10.3f"
    PSAL_ADJUSTED.FORTRAN_format = "F10.3"
    PSAL_ADJUSTED.resolution = np.float32(-0.001001001)
    PSAL_ADJUSTED[:] = final_nc_data_prof["PSAL_ADJUSTED"]

    PSAL_ADJUSTED_ERROR = nc.createVariable('PSAL_ADJUSTED_ERROR', 'f4', ('N_PROF', 'N_LEVELS'), fill_value=99999.0)
    PSAL_ADJUSTED_ERROR.long_name = "Contains the error on the adjusted values as determined by the delayed mode QC process"
    PSAL_ADJUSTED_ERROR.units = "psu"
    PSAL_ADJUSTED_ERROR.C_format = "%10.3f"
    PSAL_ADJUSTED_ERROR.FORTRAN_format = "F10.3"
    PSAL_ADJUSTED_ERROR.resolution = np.float32(-0.001001001)
    PSAL_ADJUSTED_ERROR[:] = final_nc_data_prof["PSAL_ADJUSTED_ERROR"]

    PSAL_ADJUSTED_QC = nc.createVariable('PSAL_ADJUSTED_QC', 'S1', ('N_PROF', 'N_LEVELS'), fill_value=" ")
    PSAL_ADJUSTED_QC.long_name = "quality flag"
    PSAL_ADJUSTED_QC.conventions = "Argo reference table 2"
    PSAL_ADJUSTED_QC[:] = final_nc_data_prof["PSAL_ADJUSTED_QC"].astype('S1')

    PSAL_QC = nc.createVariable('PSAL_QC', 'S1', ('N_PROF', 'N_LEVELS'), fill_value=" ")
    PSAL_QC.long_name = "quality flag"
    PSAL_QC.conventions = "Argo reference table 2"
    PSAL_QC[:] = final_nc_data_prof["PSAL_QC"].astype('S1')

    REFERENCE_DATE_TIME = nc.createVariable('REFERENCE_DATE_TIME', 'S1', ('DATE_TIME'), fill_value=" ")
    REFERENCE_DATE_TIME.long_name = "Date of reference for Julian days"
    REFERENCE_DATE_TIME.conventions = "YYYYMMDDHHMISS"
    REFERENCE_DATE_TIME[:] = np.array(list(final_nc_data_prof["REFERENCE_DATE_TIME"]), dtype='S1')

    SCIENTIFIC_CALIB_COEFFICIENT = nc.createVariable('SCIENTIFIC_CALIB_COEFFICIENT', 'S1', ('N_PROF', 'N_CALIB', 'N_PARAM', 'STRING256'), fill_value=" ")
    SCIENTIFIC_CALIB_COEFFICIENT.long_name = "Calibration coefficients for this equation"
    SCIENTIFIC_CALIB_COEFFICIENT[:] = final_nc_data_prof["SCIENTIFIC_CALIB_COEFFICIENT"]

    SCIENTIFIC_CALIB_COMMENT = nc.createVariable('SCIENTIFIC_CALIB_COMMENT', 'S1', ('N_PROF', 'N_CALIB', 'N_PARAM', 'STRING256'), fill_value=" ")
    SCIENTIFIC_CALIB_COMMENT.long_name = "Comment applying to this parameter calibration"
    SCIENTIFIC_CALIB_COMMENT[:] = final_nc_data_prof["SCIENTIFIC_CALIB_COMMENT"]

    SCIENTIFIC_CALIB_DATE = nc.createVariable('SCIENTIFIC_CALIB_DATE', 'S1', ('N_PROF', 'N_CALIB', 'N_PARAM', 'DATE_TIME'), fill_value=" ")
    SCIENTIFIC_CALIB_DATE.long_name = "Date of calibration"
    SCIENTIFIC_CALIB_DATE.conventions = "YYYYMMDDHHMISS"
    SCIENTIFIC_CALIB_DATE[:] = final_nc_data_prof["SCIENTIFIC_CALIB_DATE"]

    SCIENTIFIC_CALIB_EQUATION = nc.createVariable('SCIENTIFIC_CALIB_EQUATION', 'S1', ('N_PROF', 'N_CALIB', 'N_PARAM', 'STRING256'), fill_value=" ")
    SCIENTIFIC_CALIB_EQUATION.long_name = "Calibration equation for this parameter"
    SCIENTIFIC_CALIB_EQUATION[:] = final_nc_data_prof["SCIENTIFIC_CALIB_EQUATION"]

    STATION_PARAMETERS = nc.createVariable('STATION_PARAMETERS', 'S1', ('N_PROF', 'N_PARAM', 'STRING16'), fill_value=" ")
    STATION_PARAMETERS.long_name = "List of available parameters for the station"
    STATION_PARAMETERS.conventions = "Argo reference table 3"
    STATION_PARAMETER_temp = []
    for param in final_nc_data_prof["STATION_PARAMETERS"].split(', '):
        param_pad = np.pad(list(param), (0, 16 - len(param)), mode='constant', constant_values=' ')
        STATION_PARAMETER_temp.append(param_pad)
    STATION_PARAMETERS[:] = np.array(PARAMETER_temp, dtype='S1')   

    TEMP = nc.createVariable('TEMP', 'f4', ('N_PROF', 'N_LEVELS'), fill_value=99999.0)
    TEMP.long_name = "Sea temperature in-situ ITS-90 scale"
    TEMP.standard_name = "sea_water_temperature"
    TEMP.units = "degree_Celsius"
    TEMP.valid_min = np.float32(-2.5)
    TEMP.valid_max = np.float32(40.0)
    TEMP.C_format = "%10.3f"
    TEMP.FORTRAN_format = "F10.3"
    TEMP.resolution = np.float32(-0.001001001)
    TEMP[:] = final_nc_data_prof["TEMP"]

    TEMP_ADJUSTED = nc.createVariable('TEMP_ADJUSTED', 'f4', ('N_PROF', 'N_LEVELS'), fill_value=99999.0)
    TEMP_ADJUSTED.long_name = "Sea temperature in-situ ITS-90 scale"
    TEMP_ADJUSTED.standard_name = "sea_water_temperature"
    TEMP_ADJUSTED.units = "degree_Celsius"
    TEMP_ADJUSTED.valid_min = np.float32(-2.5)
    TEMP_ADJUSTED.valid_max = np.float32(40.0)
    TEMP_ADJUSTED.C_format = "%10.3f"
    TEMP_ADJUSTED.FORTRAN_format = "F10.3"
    TEMP_ADJUSTED.resolution = np.float32(-0.001001001)
    TEMP_ADJUSTED[:] = final_nc_data_prof["TEMP_ADJUSTED"]

    TEMP_ADJUSTED_ERROR = nc.createVariable('TEMP_ADJUSTED_ERROR', 'f4', ('N_PROF', 'N_LEVELS'), fill_value=99999.0)
    TEMP_ADJUSTED_ERROR.long_name = "Contains the error on the adjusted values as determined by the delayed mode QC process"
    TEMP_ADJUSTED_ERROR.standard_name = "sea_water_temperature"
    TEMP_ADJUSTED_ERROR.units = "degree_Celsius"
    TEMP_ADJUSTED_ERROR.C_format = "%10.3f"
    TEMP_ADJUSTED_ERROR.FORTRAN_format = "F10.3"
    TEMP_ADJUSTED_ERROR.resolution = np.float32(-0.001001001)
    TEMP_ADJUSTED_ERROR[:] = final_nc_data_prof["TEMP_ADJUSTED_ERROR"]

    TEMP_ADJUSTED_QC = nc.createVariable('TEMP_ADJUSTED_QC', 'S1', ('N_PROF', 'N_LEVELS'), fill_value=" ")
    TEMP_ADJUSTED_QC.long_name = "quality flag"
    TEMP_ADJUSTED_QC.conventions = "Argo reference table 2"
    TEMP_ADJUSTED_QC[:] = final_nc_data_prof["TEMP_ADJUSTED_QC"].astype('S1')

    TEMP_CNDC = nc.createVariable('TEMP_CNDC', 'f4', ('N_PROF', 'N_LEVELS'), fill_value= 99999.0)
    TEMP_CNDC.long_name = "Internal temperature of the conductivity cell"
    TEMP_CNDC.units = "degree_Celsius"
    TEMP_CNDC.valid_min = np.float32(-2.0)
    TEMP_CNDC.valid_max =  np.float32(40.0)
    TEMP_CNDC.C_format = "%10.3f"
    TEMP_CNDC.FORTRAN_format = "F10.3"
    TEMP_CNDC.resolution =  np.float32(0.001)
    TEMP_CNDC[:] = final_nc_data_prof["TEMP_CNDC"]

    TEMP_CNDC_QC = nc.createVariable('TEMP_CNDC_QC', 'S1', ('N_PROF', 'N_LEVELS'), fill_value= " ")
    TEMP_CNDC_QC.long_name = "quality flag"
    TEMP_CNDC_QC.conventions = "Argo reference table 2"
    TEMP_CNDC_QC[:] = final_nc_data_prof["TEMP_CNDC_QC"].astype('S1')

    TEMP_QC = nc.createVariable('TEMP_QC', 'S1', ('N_PROF', 'N_LEVELS'), fill_value= " ")
    TEMP_QC.long_name = "quality flag"
    TEMP_QC.conventions = "Argo reference table 2"
    TEMP_QC[:] = final_nc_data_prof["TEMP_QC"].astype('S1')

    VERTICAL_SAMPLING_SCHEME = nc.createVariable('VERTICAL_SAMPLING_SCHEME', 'S1', ('N_PROF', 'STRING256'), fill_value=" ")
    VERTICAL_SAMPLING_SCHEME.long_name = "Vertical sampling scheme"
    VERTICAL_SAMPLING_SCHEME.conventions = "Argo reference table 16"
    VERTICAL_SAMPLING_SCHEME[:] = np.array(np.pad(list(final_nc_data_prof["VERTICAL_SAMPLING_SCHEME"]), (0, 256 - len(final_nc_data_prof["VERTICAL_SAMPLING_SCHEME"])), mode='constant', constant_values=' '), dtype='S1')

    WMO_INST_TYPE = nc.createVariable('WMO_INST_TYPE', 'S1', ('N_PROF', 'STRING4'), fill_value=" ")
    WMO_INST_TYPE.long_name = "Type of float"
    WMO_INST_TYPE.conventions = "Argo reference table 8"
    WMO_INST_TYPE[:] = np.array(np.pad(list(final_nc_data_prof["WMO_INST_TYPE"]), (0, 4 - len(final_nc_data_prof["WMO_INST_TYPE"])), mode='constant', constant_values=' '), dtype='S1')

def calc_overall_profile_qc(qc_arr):
    """
    Calculates letter grade for parameters: PROFILE_{PARAM}_QC, based on (number of good vals)/ (total number of levels)

    Args:
        qc_arr (numpy array): qc array of associated PARAM

    Returns:
        str: letter grade of PROFILE_{PARAM}_QC, as defined in ARGO delayed mode manual 
    """

    total_levels = len(qc_arr)
    num_of_good = len(np.where((qc_arr == 1) | (qc_arr == 2))[0])

    grade = (num_of_good / total_levels) * 100
    if grade == 100:
        return 'A'
    elif grade >= 75 and grade < 100:
        return 'B'
    elif grade >= 50 and grade < 75:
        return 'C'
    elif grade >= 25 and grade < 50:
        return 'D'
    elif grade >= 0 and grade < 25:
        return 'E'
    else:
        return 'F'

def format_argo_data(argo_data):
    """
    Function to format ARGO data before delayed mode processing. 
    Procedures include:
        - for all QC arr's flip 0's to 1's
        - Set to nan, where {PARAM_ADJUSTED_QC == 4} for all PARAM_ADJUSTED arrs

    Args:
        argo_data (dict): dictionary of all associated parameters needed to generate delayed mode file.

    Returns:
        dict: argo_data reformatted 
    """

    # For all QC arr's flip 0's to 1's
    argo_data["JULD_QC"][argo_data["JULD_QC"] == 0] = 1
    argo_data["JULD_QC"][np.isnan(argo_data["JULD_QC"])] = 1
    argo_data["POSITION_QC"][argo_data["POSITION_QC"] == 0] = 1
    argo_data["POSITION_QC"][np.isnan(argo_data["POSITION_QC"])] = 1
    argo_data["PRES_ADJUSTED_QC"][argo_data["PRES_ADJUSTED_QC"] == 0] = 1
    argo_data["PSAL_ADJUSTED_QC"][argo_data["PSAL_ADJUSTED_QC"] == 0] = 1
    argo_data["TEMP_ADJUSTED_QC"][argo_data["TEMP_ADJUSTED_QC"] == 0] = 1
    argo_data["TEMP_CNDC_QC"][argo_data["TEMP_CNDC_QC"] == 0] = 1
    argo_data["NB_SAMPLE_CTD_QC"][argo_data["NB_SAMPLE_CTD_QC"] == 0] = 1

    # Set to nan, where {PARAM_ADJUSTED_QC == 4} 
    argo_data["PRES_ADJUSTED"][argo_data["PRES_ADJUSTED_QC"] == 4] = np.nan
    argo_data["PSAL_ADJUSTED"][argo_data["PSAL_ADJUSTED_QC"] == 4] = np.nan
    argo_data["TEMP_ADJUSTED"][argo_data["TEMP_ADJUSTED_QC"] == 4] = np.nan
    argo_data["CNDC_ADJUSTED"][argo_data["CNDC_ADJUSTED_QC"] == 4] = np.nan

    return argo_data

def fill_other_history_parem_arrs(final_nc_data_prof, parems_to_fill = None):
    """_summary_

    Args:
        final_nc_data_prof (dict): dictionary of all associated parameters needed to generate file.
        parems_to_fill (list of str, optional): List of HISTORY_{PARAM}s to fill that are not always set. Defaults to None.

    Returns:
        dict: returns final_nc_data_prof with set HISTORY_{PARAM} values
    """
    
    # PAREMS that are always set
    if final_nc_data_prof["HISTORY_DATE"] is None:
        final_nc_data_prof["HISTORY_DATE"] = np.array(list(str(datetime.now().strftime("%Y%m%d%H%M%S"))))
    else:
        final_nc_data_prof["HISTORY_DATE"] = np.vstack([final_nc_data_prof["HISTORY_DATE"], np.array(list(str(datetime.now().strftime("%Y%m%d%H%M%S"))))])
    
    if final_nc_data_prof["HISTORY_INSTITUTION"] is None:
        final_nc_data_prof["HISTORY_INSTITUTION"] = np.array(list('JP  '))
    else:
        final_nc_data_prof["HISTORY_INSTITUTION"] = np.vstack([final_nc_data_prof["HISTORY_INSTITUTION"], np.array(list('JP  '))])
    
    if final_nc_data_prof["HISTORY_STEP"] is None:
        final_nc_data_prof["HISTORY_STEP"] = np.array(list('ARSQ'))
    else:
        final_nc_data_prof["HISTORY_STEP"] = np.vstack([final_nc_data_prof["HISTORY_STEP"], np.array(list('ARSQ'))])
    
    if final_nc_data_prof['HISTORY_SOFTWARE'] is None:
        final_nc_data_prof['HISTORY_SOFTWARE'] =  np.array(list('DMPS'))
    else:
        final_nc_data_prof['HISTORY_SOFTWARE'] = np.vstack([final_nc_data_prof["HISTORY_SOFTWARE"], np.array(list('DMPS'))])
    
    if final_nc_data_prof['HISTORY_SOFTWARE_RELEASE'] is None:
        final_nc_data_prof['HISTORY_SOFTWARE_RELEASE'] = np.array(list('B_V0'))
    else:
        final_nc_data_prof['HISTORY_SOFTWARE_RELEASE'] = np.vstack([final_nc_data_prof["HISTORY_SOFTWARE_RELEASE"], np.array(list('B_V0'))])

    # PAREMS that are sometimes set
    # if these PAREMS are not set, we fill the data w/ empty vals
    if parems_to_fill is not None:
        for parem in parems_to_fill:
            if parem == 'HISTORY_START_PRES':
                if final_nc_data_prof['HISTORY_START_PRES'] is None:
                    final_nc_data_prof['HISTORY_START_PRES'] = 99999.0
                else:
                    final_nc_data_prof['HISTORY_START_PRES'] = np.append(final_nc_data_prof["HISTORY_START_PRES"], 99999.0)
            elif parem == 'HISTORY_STOP_PRES':
                if final_nc_data_prof['HISTORY_STOP_PRES'] is None:
                    final_nc_data_prof['HISTORY_STOP_PRES'] = 99999.0
                else:
                    final_nc_data_prof['HISTORY_STOP_PRES'] = np.append(final_nc_data_prof["HISTORY_STOP_PRES"], 99999.0)
            elif parem == 'HISTORY_QCTEST':
                if final_nc_data_prof['HISTORY_QCTEST'] is None:
                    final_nc_data_prof['HISTORY_QCTEST'] = np.array(list(' ' * 16))
                else:
                    final_nc_data_prof['HISTORY_QCTEST'] = np.vstack([final_nc_data_prof["HISTORY_QCTEST"], np.array(list(' ' * 16))])
            elif parem == 'HISTORY_PARAMETER':
                if final_nc_data_prof['HISTORY_PARAMETER'] is None:
                    final_nc_data_prof['HISTORY_PARAMETER'] =  np.array(list(' ' * 16))
                else:
                    final_nc_data_prof['HISTORY_PARAMETER'] = np.vstack([final_nc_data_prof["HISTORY_PARAMETER"], np.array(list(' ' * 16))])
            elif parem == 'HISTORY_PREVIOUS_VALUE':
                if final_nc_data_prof['HISTORY_PREVIOUS_VALUE'] is None:
                    final_nc_data_prof['HISTORY_PREVIOUS_VALUE'] = 99999.0
                else:
                    final_nc_data_prof['HISTORY_PREVIOUS_VALUE'] = np.append(final_nc_data_prof["HISTORY_PREVIOUS_VALUE"], 99999.0)
            elif parem == 'HISTORY_REFERENCE':
                if final_nc_data_prof['HISTORY_REFERENCE'] is None:
                    final_nc_data_prof['HISTORY_REFERENCE'] = np.array(list(' ' * 64))
                else:
                    final_nc_data_prof['HISTORY_REFERENCE'] = np.vstack([final_nc_data_prof["HISTORY_REFERENCE"], np.array(list(' ' * 64))])
    
    return final_nc_data_prof

def set_history_parems(final_nc_data_prof, type_to_set, **kwargs):
    """
    This function sets HISTORY_{PARAM} for ARGO delayed mode processing NETCDF files.

    Args:
        final_nc_data_prof (dict): dictionary of all associated parameters needed to generate file.
        type_to_set (str): Associated HISTORY_{PARAM} to set

    Returns:
        dict: returns final_nc_data_prof with set HISTORY_{PARAM} values
    """
    
    if type_to_set == "SET_IP":
        # We always set this first
        if final_nc_data_prof["HISTORY_ACTION"] is None:
            final_nc_data_prof["HISTORY_ACTION"] = np.array(list('  IP'))
        else:
            final_nc_data_prof["HISTORY_ACTION"] = np.vstack([final_nc_data_prof["HISTORY_ACTION"], np.array(list('  IP'))])
        
        final_nc_data_prof = fill_other_history_parem_arrs(final_nc_data_prof, ['HISTORY_START_PRES', 'HISTORY_STOP_PRES', 'HISTORY_QCTEST', 'HISTORY_PARAMETER', 'HISTORY_PREVIOUS_VALUE', 'HISTORY_REFERENCE'])
    
    # So no need to verify if arrs are empty for remaining if statements
    if type_to_set == "SET_QCP$" or "SET_QCF$":
        if type_to_set == "SET_QCF$":
            final_nc_data_prof["HISTORY_ACTION"] = np.vstack([final_nc_data_prof["HISTORY_ACTION"], np.array(list('QCF$'))])
        if type_to_set == "SET_QCP$":
            final_nc_data_prof["HISTORY_ACTION"] = np.vstack([final_nc_data_prof["HISTORY_ACTION"], np.array(list('QCP$'))])

        if 'HISTORY_QCTEST_names' in kwargs:
            hex_num = history_qc_test_converter(kwargs['HISTORY_QCTEST_names'], mode='encode')
            padded_hex = np.pad(list(hex_num), (0, 16 - len(hex_num)), mode='constant', constant_values=' ')
            final_nc_data_prof["HISTORY_QCTEST"] = np.vstack([final_nc_data_prof["HISTORY_QCTEST"], padded_hex])
            final_nc_data_prof = fill_other_history_parem_arrs(final_nc_data_prof, ['HISTORY_START_PRES', 'HISTORY_STOP_PRES', 'HISTORY_PARAMETER', 'HISTORY_PREVIOUS_VALUE', 'HISTORY_REFERENCE'])
        else:
            final_nc_data_prof = fill_other_history_parem_arrs(final_nc_data_prof, ['HISTORY_START_PRES', 'HISTORY_STOP_PRES', 'HISTORY_QCTEST', 'HISTORY_PARAMETER', 'HISTORY_PREVIOUS_VALUE', 'HISTORY_REFERENCE'])

    if type_to_set == "SET_CF":
        final_nc_data_prof["HISTORY_ACTION"] = np.vstack([final_nc_data_prof["HISTORY_ACTION"], np.array(list('  CF'))])

        if 'parems_set' in kwargs:
            parems_as_str = ""
            for a in kwargs['parems_set']:
                parems_as_str = parems_as_str + a + ","
            padded_parem = np.pad(list(parems_as_str[:-1]), (0, 16 - len(parems_as_str[:-1])), mode='constant', constant_values=' ')
            final_nc_data_prof["HISTORY_PARAMETER"] = np.vstack([final_nc_data_prof["HISTORY_PARAMETER"], padded_parem])
            final_nc_data_prof = fill_other_history_parem_arrs(final_nc_data_prof, ['HISTORY_START_PRES', 'HISTORY_STOP_PRES', 'HISTORY_QCTEST', 'HISTORY_PREVIOUS_VALUE', 'HISTORY_REFERENCE'])
        else:
            final_nc_data_prof = fill_other_history_parem_arrs(final_nc_data_prof, ['HISTORY_START_PRES', 'HISTORY_STOP_PRES', 'HISTORY_QCTEST', 'HISTORY_PARAMETER', 'HISTORY_PREVIOUS_VALUE', 'HISTORY_REFERENCE'])

    if type_to_set == "SET_POS_INTERP" or type_to_set == "SET_JULD_INTERP":
        final_nc_data_prof["HISTORY_ACTION"] = np.vstack([final_nc_data_prof["HISTORY_ACTION"], np.array(list('  CV'))])
        
        if type_to_set == "SET_POS_INTERP":
            parems_as_str = "LAT$,LON$"
            padded_parem = np.pad(list(parems_as_str), (0, 16 - len(parems_as_str)), mode='constant', constant_values=' ')
            final_nc_data_prof["HISTORY_PARAMETER"] = np.vstack([final_nc_data_prof["HISTORY_PARAMETER"], padded_parem])
    
        if type_to_set == "SET_JULD_INTERP":
            parems_as_str = "JULD"
            padded_parem = np.pad(list(parems_as_str), (0, 16 - len(parems_as_str)), mode='constant', constant_values=' ')
            final_nc_data_prof["HISTORY_PARAMETER"] = np.vstack([final_nc_data_prof["HISTORY_PARAMETER"], padded_parem])
    
        final_nc_data_prof = fill_other_history_parem_arrs(final_nc_data_prof, ['HISTORY_START_PRES', 'HISTORY_STOP_PRES', 'HISTORY_QCTEST', 'HISTORY_PREVIOUS_VALUE', 'HISTORY_REFERENCE'])
    
    return final_nc_data_prof

def get_padded_array(kwargs, key, pad_type):

    value = kwargs.get(key)

    if value is None:
        if pad_type == "SET_DATE":
            return np.pad(list(' '), (0, 14 - 1), mode='constant', constant_values=' ')
        else:
            return np.pad(list('none'), (0, 256 - 4), mode='constant', constant_values=' ')
    else:
        if pad_type == "SET_DATE":
            return np.pad(
                list(value),
                (0, 14 - len(value)),
                mode='constant',
                constant_values=' '
            )
        else:
            return np.pad(
                list(value),
                (0, 256 - len(value)),
                mode='constant',
                constant_values=' '
            )
    
def set_sci_calib_parems(final_nc_data_prof, param_to_set, **kwargs):

    pres = get_padded_array(kwargs, "pres", param_to_set)
    temp = get_padded_array(kwargs, "temp", param_to_set)
    cndc = get_padded_array(kwargs, "cndc", param_to_set)
    psal = get_padded_array(kwargs, "psal", param_to_set)
    temp_cndc = get_padded_array(kwargs, "temp_cndc", param_to_set)
    nb_sample_ctd = get_padded_array(kwargs, "nb_sample_ctd", param_to_set)

    if param_to_set == "SET_COEFFICIENT":
        if final_nc_data_prof["SCIENTIFIC_CALIB_COEFFICIENT"] is None:
            final_nc_data_prof["SCIENTIFIC_CALIB_COEFFICIENT"] = np.stack([pres, temp, cndc, psal, temp_cndc, nb_sample_ctd])
        else:
            final_nc_data_prof["SCIENTIFIC_CALIB_COEFFICIENT"] = np.concatenate([final_nc_data_prof["SCIENTIFIC_CALIB_COEFFICIENT"], np.stack([pres, temp, cndc, psal, temp_cndc, nb_sample_ctd])[np.newaxis, ...]], axis=0)
    if param_to_set == "SET_COMMENT":
        if final_nc_data_prof["SCIENTIFIC_CALIB_COMMENT"] is None:
            final_nc_data_prof["SCIENTIFIC_CALIB_COMMENT"] = np.stack([pres, temp, cndc, psal, temp_cndc, nb_sample_ctd])
        else:
            final_nc_data_prof["SCIENTIFIC_CALIB_COMMENT"] = np.concatenate([final_nc_data_prof["SCIENTIFIC_CALIB_COMMENT"], np.stack([pres, temp, cndc, psal, temp_cndc, nb_sample_ctd])[np.newaxis, ...]], axis=0)
    if param_to_set == "SET_EQUATION":
        if final_nc_data_prof["SCIENTIFIC_CALIB_EQUATION"] is None:
            final_nc_data_prof["SCIENTIFIC_CALIB_EQUATION"] = np.stack([pres, temp, cndc, psal, temp_cndc, nb_sample_ctd])
        else:
            final_nc_data_prof["SCIENTIFIC_CALIB_EQUATION"] = np.concatenate([final_nc_data_prof["SCIENTIFIC_CALIB_EQUATION"], np.stack([pres, temp, cndc, psal, temp_cndc, nb_sample_ctd])[np.newaxis, ...]], axis=0)
    if param_to_set == "SET_DATE":
        if final_nc_data_prof["SCIENTIFIC_CALIB_DATE"] is None:
            final_nc_data_prof["SCIENTIFIC_CALIB_DATE"] = np.stack([pres, temp, cndc, psal, temp_cndc, nb_sample_ctd])
        else:
            final_nc_data_prof["SCIENTIFIC_CALIB_DATE"] =  np.concatenate([final_nc_data_prof["SCIENTIFIC_CALIB_DATE"], np.stack([pres, temp, cndc, psal, temp_cndc, nb_sample_ctd])[np.newaxis, ...]], axis=0)
    
    return final_nc_data_prof

        
def sal_recalc_RBRargo3_3k_procedures(processed_argo_data):

    # Step 1: recompute sal due to compressinility effect  
    Co = gsw.C_from_SP(processed_argo_data["PSALs"], processed_argo_data["TEMPs"], processed_argo_data["PRESs"])
    PSAL_ADJUSTED_Padj = gsw.SP_from_C(Co, processed_argo_data["TEMP_ADJUSTED"], processed_argo_data["PRES_ADJUSTED"])

    # Step 2: apply thermal inertia correction 
    # a) check TEMP_CNDC visually 
    Cadj = gsw.C_from_SP(PSAL_ADJUSTED_Padj, processed_argo_data["TEMP_ADJUSTED"], processed_argo_data["PRES_ADJUSTED"])
    # c) estimate elapsed time
    # d) compute TEMP_celltm
    
def process_data_dmode_files(nc_filepath, float_num, dest_filepath, config_fp, org_netcdf_fp = None):

    if org_netcdf_fp is not None:
        org_files = sorted(glob.glob(os.path.join(org_netcdf_fp, "*.nc")))
    processed_argo_data = read_intermediate_nc_file(nc_filepath)
    processed_argo_data = format_argo_data(processed_argo_data)

    # init fillvals for ADJUSTED_ERROR_PAREM
    CNDC_ADJUSTED_ERROR_FILLVAL = np.nan
    PRES_ADJUSTED_ERROR_FILLVAL = np.nan
    PSAL_ADJUSTED_ERROR_FILLVAL = np.nan
    TEMP_ADJUSTED_ERROR_FILLVAL = np.nan

    # Initialize dictionary with 'noval' for all keys
    final_nc_data = {
        'CNDC': 'noval', 
        'CNDC_ADJUSTED': 'noval', 
        'CNDC_ADJUSTED_ERROR': 'noval', 
        'CNDC_ADJUSTED_QC': 'noval', 
        'CNDC_QC': 'noval', 
        'CONFIG_MISSION_NUMBER': 'noval',  
        'CYCLE_NUMBER': 'noval',
        'DATA_CENTRE': 'noval',
        'DATA_MODE': 'noval',
        'DATA_STATE_INDICTATOR': 'noval',
        'DATA_TYPE': 'noval',
        'DATE_CREATION': 'noval',
        'DATE_UPDATE': 'noval',
        'DC_REFERENCE': 'noval',
        'DIRECTION': 'noval',
        'FIRMWARE_VERSION': 'noval',
        'FLOAT_SERIAL_NO': 'noval',
        'FORMAT_VERSION': 'noval',
        'HANDBOOK_VERSION': 'noval',
        'HISTORY_ACTION': 'noval',
        'HISTORY_DATE': 'noval',
        'HISTORY_INSITUTION': 'noval',
        'HISTORY_PARAMETER': 'noval',
        'HISTORY_PREVIOUS_VALUE': 'noval',
        'HISTORY_QCTEST': 'noval',
        'HISTORY_REFERENCE': 'noval',
        'HISTORY_SOFTWARE': 'noval',
        'HISTORY_SOFTWARE_RELEASE': 'noval',
        'HISTORY_START_PRES': 'noval',
        'HISTORY_STEP': 'noval',
        'HISTORY_STOP_PRES': 'noval',
        'JULD': 'noval',
        'JULD_LOCATION': 'noval',
        'JULD_QC': 'noval',
        'LATITUDE': 'noval',
        'LONGITUDE': 'noval',
        'NB_SAMPLE_CTD': 'noval',
        'NB_SAMPLE_CTD_QC': 'noval',
        'PARAMETER': 'noval',
        'PI_NAME': 'noval',
        'PLATFORM NUMBER': 'noval',
        'PLATFORM_TYPE': 'noval',
        'POSITION_QC': 'noval',
        'POSITIONING_SYSTEM': 'noval',
        'PRES': 'noval',
        'PRES_ADJUSTED': 'noval',
        'PRES_ADJUSTED_ERROR': 'noval',
        'PRES_ADJUSTED_QC': 'noval',
        'PRES_QC': 'noval',
        'PROFILE_CNDC_QC': 'noval',
        'PROFILE_NB_SAMPLE_CTD_QC': 'noval',
        'PROFILE_PRES_QC': 'noval',
        'PROFILE_PSAL_QC': 'noval',
        'PROFILE_TEMP_CNDC_QC': 'noval',
        'PROFILE_TEMP_QC': 'noval',
        'PROJECT_NAME': 'noval',
        'PSAL': 'noval',
        'PSAL_ADJUSTED': 'noval',
        'PSAL_ADJUSTED_ERROR': 'noval',
        'PSAL_ADJUSTED_QC': 'noval',
        'PSAL_QC': 'noval',
        'REFERENCE_DATE_TIME': 'noval',
        'SCIENTIFIC_CALIB_COEFFICIENT': 'noval',
        'SCIENTIFIC_CALIB_COMMENT': 'noval',
        'SCIENTIFIC_CALIB_DATE': 'noval',
        'SCIENTIFIC_CALIB_EQUATION': 'noval',
        'STATION_PARAMETERS': 'noval',
        'TEMP': 'noval',
        'TEMP_ADJUSTED': 'noval',
        'TEMP_ADJUSTED_ERROR': 'noval',
        'TEMP_ADJUSTED_QC': 'noval',
        'TEMP_CNDC': 'noval',
        'TEMP_CNDC_QC': 'noval',
        'TEMP_QC': 'noval',
        'VERTICAL_SAMPLING_SCHEME': 'noval',
        'WMO_INST_TYPE': 'noval',
    }

    # Read config file info  
    start_of_file = False
    with open(config_fp, 'r') as config_file:
        for line in config_file:
            line = line.strip()  # Remove leading/trailing whitespaces
            
            if start_of_file == True:  
                param_val = line.split('=', 1)[1].strip()
            if start_of_file == True and param_val == "None":
                raise Exception("Please make sure all fields in config file are filled! ")

            if 'DATA_CENTRE' in line:
                final_nc_data["DATA_CENTRE"] = param_val
            elif 'DATA_MODE' in line:
                final_nc_data["DATA_MODE"] = param_val
            elif 'DATA_STATE_INDICATOR' in line:
                final_nc_data["DATA_STATE_INDICTATOR"] = param_val
            elif 'DATA_TYPE' in line:
                final_nc_data["DATA_TYPE"] = param_val
            elif 'DC_REFERENCE' in line:
                final_nc_data["DC_REFERENCE"] = param_val
            elif 'DIRECTION' in line:
                final_nc_data["DIRECTION"] = param_val
            elif 'FIRMWARE_VERSION' in line:
                final_nc_data["FIRMWARE_VERSION"] = param_val
            elif 'FLOAT_SERIAL_NO' in line:
                final_nc_data["FLOAT_SERIAL_NO"] = param_val
            elif 'FORMAT_VERSION' in line:
                final_nc_data["FORMAT_VERSION"] = param_val
            elif 'HANDBOOK_VERSION' in line:
                final_nc_data["HANDBOOK_VERSION"] = param_val
            elif 'PI_NAME' in line:
                final_nc_data["PI_NAME"] = param_val
            elif 'PLATFORM_NUMBER' in line:
                final_nc_data["PLATFORM_NUMBER"] = param_val
            elif 'PLATFORM_TYPE' in line:
                final_nc_data["PLATFORM_TYPE"] = param_val
            elif 'POSITIONING_SYSTEM' in line:
                final_nc_data["POSITIONING_SYSTEM"] = param_val
            elif 'PROJECT_NAME' in line:
                final_nc_data["PROJECT_NAME"] = param_val
            elif 'REFERENCE_DATE_TIME' in line:
                final_nc_data["REFERENCE_DATE_TIME"] = param_val
            elif 'STATION_PARAMETERS' in line:
                final_nc_data["STATION_PARAMETERS"] = param_val
            elif 'PARAMETER' in line:
                final_nc_data["PARAMETER"] = param_val
            elif 'VERTICAL_SAMPLING_SCHEME' in line:
                final_nc_data["VERTICAL_SAMPLING_SCHEME"] = param_val
            elif 'WMO_INST_TYPE' in line:
                final_nc_data["WMO_INST_TYPE"] = param_val
            elif 'CNDC_ADJUSTED_ERROR' in line:
                CNDC_ADJUSTED_ERROR_FILLVAL = float(param_val)
            elif 'PRES_ADJUSTED_ERROR' in line:
                PRES_ADJUSTED_ERROR_FILLVAL = float(param_val)
            elif 'PSAL_ADJUSTED_ERROR' in line:
                PSAL_ADJUSTED_ERROR_FILLVAL = float(param_val)
            elif 'TEMP_ADJUSTED_ERROR' in line:
                TEMP_ADJUSTED_ERROR_FILLVAL = float(param_val)
            if line == "===========================================================================":
                start_of_file = True

    for profile_num in processed_argo_data["PROFILE_NUMS"]:

        if org_netcdf_fp is not None:
            org_profile_file = [f for f in org_files if f.endswith(f"R{float_num}_{profile_num:03}.nc")]
            argo_org_file = nc4.Dataset(org_profile_file[0])

        # Get corresponding index of profile in argo_data dict
        i = np.where(processed_argo_data["PROFILE_NUMS"] == profile_num)
        # Make deep copy of dict to edit for each profile
        final_nc_data_prof = copy.deepcopy(final_nc_data)
        # Get index to remove traling NaNs
        nan_index = np.where(~np.isnan(np.squeeze(processed_argo_data["PRESs"][i, :])))[0][-1] + 1

        # init vars of final_nc_data
        final_nc_data_prof["CNDC"] = np.squeeze(processed_argo_data["CNDCs"][i, :nan_index])
        final_nc_data_prof["CNDC_ADJUSTED"] = np.squeeze(processed_argo_data["CNDC_ADJUSTED"][i, :nan_index])
        final_nc_data_prof["CNDC_ADJUSTED_QC"] = np.squeeze(processed_argo_data["CNDC_ADJUSTED_QC"][i, :nan_index])
        final_nc_data_prof["CNDC_QC"] = np.squeeze(processed_argo_data["CNDC_QC"][i, :nan_index])

        final_nc_data_prof["CONFIG_MISSION_NUMBER"] = profile_num
        final_nc_data_prof["CYCLE_NUMBER"] = profile_num
        
        final_nc_data_prof["DATE_UPDATE"] = list(str(datetime.now().strftime("%Y%m%d%H%M%S")))
        if org_netcdf_fp is None:
            final_nc_data_prof["DATE_CREATION"] = list(str(datetime.now().strftime("%Y%m%d%H%M%S")))
        else:
            final_nc_data_prof["DATE_CREATION"] = np.squeeze(argo_org_file.variables["DATE_CREATION"][:].filled(argo_org_file.variables["DATE_CREATION"].getncattr("_FillValue")))

        final_nc_data_prof["JULD"] = np.squeeze(processed_argo_data["JULDs"][i])
        final_nc_data_prof["JULD_LOCATION"] = np.squeeze(processed_argo_data["JULD_LOCATIONs"][i])
        final_nc_data_prof["JULD_QC"] = np.squeeze(processed_argo_data["JULD_QC"][i])
        final_nc_data_prof["LATITUDE"] = np.squeeze(processed_argo_data["LATs"][i])
        final_nc_data_prof["LONGITUDE"] = np.squeeze(processed_argo_data["LONs"][i])
        final_nc_data_prof["NB_SAMPLE_CTD"] = np.squeeze(processed_argo_data["NB_SAMPLE_CTD"][i, :nan_index])
        final_nc_data_prof["NB_SAMPLE_CTD_QC"] = np.squeeze(processed_argo_data["NB_SAMPLE_CTD_QC"][i, :nan_index])
        final_nc_data_prof["POSITION_QC"] = np.squeeze(processed_argo_data["POSITION_QC"][i])

        final_nc_data_prof["PRES"] = np.squeeze(processed_argo_data["PRESs"][i, :nan_index])
        final_nc_data_prof["PRES_ADJUSTED"] = np.squeeze(processed_argo_data["PRES_ADJUSTED"][i, :nan_index])
        final_nc_data_prof["PRES_ADJUSTED_QC"] = np.squeeze(processed_argo_data["PRES_ADJUSTED_QC"][i, :nan_index])
        final_nc_data_prof["PRES_QC"] = np.squeeze(processed_argo_data["PRES_QC"][i, :nan_index])

        final_nc_data_prof["TEMP"] = np.squeeze(processed_argo_data["TEMPs"][i, :nan_index])
        final_nc_data_prof["TEMP_ADJUSTED"] = np.squeeze(processed_argo_data["TEMP_ADJUSTED"][i, :nan_index])
        final_nc_data_prof["TEMP_ADJUSTED_QC"] = np.squeeze(processed_argo_data["TEMP_ADJUSTED_QC"][i, :nan_index])
        final_nc_data_prof["TEMP_QC"] = np.squeeze(processed_argo_data["TEMP_QC"][i, :nan_index])
        final_nc_data_prof["TEMP_CNDC"] = np.squeeze(processed_argo_data["TEMP_CNDCs"][i, :nan_index])
        final_nc_data_prof["TEMP_CNDC_QC"] = np.squeeze(processed_argo_data["TEMP_CNDC_QC"][i, :nan_index])
        
        final_nc_data_prof["PSAL"] = np.squeeze(processed_argo_data["PSALs"][i, :nan_index])
        final_nc_data_prof["PSAL_ADJUSTED"] = np.squeeze(processed_argo_data["PSAL_ADJUSTED"][i, :nan_index])
        final_nc_data_prof["PSAL_ADJUSTED_QC"] = np.squeeze(processed_argo_data["PSAL_ADJUSTED_QC"][i, :nan_index])
        final_nc_data_prof["PSAL_QC"] = np.squeeze(processed_argo_data["PSAL_QC"][i, :nan_index])

        # Set {PAREM}_ADJUSTED_ERROR arrs
        final_nc_data_prof["CNDC_ADJUSTED_ERROR"] = np.full(np.squeeze(processed_argo_data["TEMP_ADJUSTED"][i, :nan_index]).shape, fill_value = CNDC_ADJUSTED_ERROR_FILLVAL)
        final_nc_data_prof["PRES_ADJUSTED_ERROR"] = np.full(np.squeeze(processed_argo_data["TEMP_ADJUSTED"][i, :nan_index]).shape, fill_value = PRES_ADJUSTED_ERROR_FILLVAL)
        final_nc_data_prof["PSAL_ADJUSTED_ERROR"] = np.full(np.squeeze(processed_argo_data["TEMP_ADJUSTED"][i, :nan_index]).shape, fill_value = PSAL_ADJUSTED_ERROR_FILLVAL)
        final_nc_data_prof["TEMP_ADJUSTED_ERROR"] = np.full(np.squeeze(processed_argo_data["TEMP_ADJUSTED"][i, :nan_index]).shape, fill_value = TEMP_ADJUSTED_ERROR_FILLVAL)
        
        # Set overall profile quality flag
        final_nc_data_prof["PROFILE_CNDC_QC"] = calc_overall_profile_qc(np.squeeze(processed_argo_data["PSAL_ADJUSTED_QC"][i, :nan_index]))    # Because we replace CNDC QC data arrs with PSAL QC ones
        final_nc_data_prof["PROFILE_NB_SAMPLE_CTD_QC"] = calc_overall_profile_qc(np.squeeze(processed_argo_data["NB_SAMPLE_CTD_QC"][i, :nan_index]))
        final_nc_data_prof["PROFILE_PRES_QC"] = calc_overall_profile_qc(np.squeeze(processed_argo_data["PRES_ADJUSTED_QC"][i, :nan_index]))
        final_nc_data_prof["PROFILE_PSAL_QC"] = calc_overall_profile_qc(np.squeeze(processed_argo_data["PSAL_ADJUSTED_QC"][i, :nan_index]))
        final_nc_data_prof["PROFILE_TEMP_CNDC_QC"] = calc_overall_profile_qc(np.squeeze(processed_argo_data["TEMP_CNDC_QC"][i, :nan_index]))
        final_nc_data_prof["PROFILE_TEMP_QC"] = calc_overall_profile_qc(np.squeeze(processed_argo_data["TEMP_ADJUSTED_QC"][i, :nan_index]))

        # Set scientific_calib parems
        # Get org data if it exists 
        if org_netcdf_fp is not None:
            final_nc_data_prof["SCIENTIFIC_CALIB_COEFFICIENT"] = np.squeeze(np.char.decode(argo_org_file.variables["SCIENTIFIC_CALIB_COEFFICIENT"][:].filled(argo_org_file.variables["SCIENTIFIC_CALIB_COEFFICIENT"].getncattr("_FillValue"))))
            final_nc_data_prof["SCIENTIFIC_CALIB_COMMENT"] = np.squeeze(np.char.decode(argo_org_file.variables["SCIENTIFIC_CALIB_COMMENT"][:].filled(argo_org_file.variables["SCIENTIFIC_CALIB_COMMENT"].getncattr("_FillValue"))))
            final_nc_data_prof["SCIENTIFIC_CALIB_DATE"] = np.squeeze(np.char.decode(argo_org_file.variables["SCIENTIFIC_CALIB_DATE"][:].filled(argo_org_file.variables["SCIENTIFIC_CALIB_DATE"].getncattr("_FillValue"))))
            final_nc_data_prof["SCIENTIFIC_CALIB_EQUATION"] = np.squeeze(np.char.decode(argo_org_file.variables["SCIENTIFIC_CALIB_EQUATION"][:].filled(argo_org_file.variables["SCIENTIFIC_CALIB_EQUATION"].getncattr("_FillValue"))))
        
        else:
            final_nc_data_prof["SCIENTIFIC_CALIB_COEFFICIENT"] = None
            final_nc_data_prof["SCIENTIFIC_CALIB_COMMENT"] = None
            final_nc_data_prof["SCIENTIFIC_CALIB_DATE"] = None
            final_nc_data_prof["SCIENTIFIC_CALIB_EQUATION"] = None

        if processed_argo_data["PRES_OFFSET"][i][0] is not None:
            set_sci_calib_parems(final_nc_data_prof, "SET_COEFFICIENT", 
                                    pres = f"surface_pressure={float(processed_argo_data["PRES_OFFSET"][i][0]):.2f} dbar")
        else:
            set_sci_calib_parems(final_nc_data_prof, "SET_COEFFICIENT")
        
        set_sci_calib_parems(final_nc_data_prof, "SET_COMMENT",
                                pres = f"Pressure adjusted during delayed mode processing based on most recent valid surface pressure")
        set_sci_calib_parems(final_nc_data_prof, "SET_EQUATION",
                                pres = f"PRES_ADJUSTED = PRES - surface_pressure")
        set_sci_calib_parems(final_nc_data_prof, "SET_DATE",
                                pres = f"{str(datetime.now().strftime("%Y%m%d%H%M%S"))}")
        
        # expand dim if it is incorrect
        if len(final_nc_data_prof["SCIENTIFIC_CALIB_COEFFICIENT"].shape) == 2:
            final_nc_data_prof["SCIENTIFIC_CALIB_COEFFICIENT"] = np.expand_dims(final_nc_data_prof["SCIENTIFIC_CALIB_COEFFICIENT"], axis=0)
            final_nc_data_prof["SCIENTIFIC_CALIB_COMMENT"] = np.expand_dims(final_nc_data_prof["SCIENTIFIC_CALIB_COMMENT"], axis=0)
            final_nc_data_prof["SCIENTIFIC_CALIB_DATE"] = np.expand_dims(final_nc_data_prof["SCIENTIFIC_CALIB_DATE"], axis=0)
            final_nc_data_prof["SCIENTIFIC_CALIB_EQUATION"] = np.expand_dims(final_nc_data_prof["SCIENTIFIC_CALIB_EQUATION"], axis=0)

        # Get history params
        # Get org data if it exists 
        if org_netcdf_fp is not None:
            final_nc_data_prof["HISTORY_SOFTWARE"] = np.squeeze(np.char.decode(argo_org_file.variables["HISTORY_SOFTWARE"][:].filled(argo_org_file.variables["HISTORY_SOFTWARE"].getncattr("_FillValue"))))
            final_nc_data_prof["HISTORY_SOFTWARE_RELEASE"] = np.squeeze(np.char.decode(argo_org_file.variables["HISTORY_SOFTWARE_RELEASE"][:].filled(argo_org_file.variables["HISTORY_SOFTWARE_RELEASE"].getncattr("_FillValue"))))
            final_nc_data_prof["HISTORY_REFERENCE"] = np.squeeze(np.char.decode(argo_org_file.variables["HISTORY_REFERENCE"][:].filled(argo_org_file.variables["HISTORY_REFERENCE"].getncattr("_FillValue"))))
            final_nc_data_prof["HISTORY_START_PRES"] = np.squeeze(argo_org_file.variables["HISTORY_START_PRES"][:].filled(argo_org_file.variables["HISTORY_START_PRES"].getncattr("_FillValue")))
            final_nc_data_prof["HISTORY_STOP_PRES"] = np.squeeze(argo_org_file.variables["HISTORY_STOP_PRES"][:].filled(argo_org_file.variables["HISTORY_STOP_PRES"].getncattr("_FillValue")))
            final_nc_data_prof["HISTORY_ACTION"] = np.squeeze(np.char.decode(argo_org_file.variables["HISTORY_ACTION"][:].filled(argo_org_file.variables["HISTORY_ACTION"].getncattr("_FillValue"))))
            final_nc_data_prof["HISTORY_QCTEST"] = np.squeeze(np.char.decode(argo_org_file.variables["HISTORY_QCTEST"][:].filled(argo_org_file.variables["HISTORY_QCTEST"].getncattr("_FillValue"))))
            final_nc_data_prof["HISTORY_PARAMETER"] = np.squeeze(np.char.decode(argo_org_file.variables["HISTORY_PARAMETER"][:].filled(argo_org_file.variables["HISTORY_PARAMETER"].getncattr("_FillValue"))))
            final_nc_data_prof["HISTORY_STEP"] = np.squeeze(np.char.decode(argo_org_file.variables["HISTORY_STEP"][:].filled(argo_org_file.variables["HISTORY_STEP"].getncattr("_FillValue"))))
            final_nc_data_prof["HISTORY_PREVIOUS_VALUE"] = np.squeeze(argo_org_file.variables["HISTORY_PREVIOUS_VALUE"][:].filled(argo_org_file.variables["HISTORY_PREVIOUS_VALUE"].getncattr("_FillValue")))
            final_nc_data_prof["HISTORY_DATE"] = np.squeeze(np.char.decode(argo_org_file.variables["HISTORY_DATE"][:].filled(argo_org_file.variables["HISTORY_DATE"].getncattr("_FillValue"))))
            final_nc_data_prof["HISTORY_INSTITUTION"] = np.squeeze(np.char.decode(argo_org_file.variables["HISTORY_INSTITUTION"][:].filled(argo_org_file.variables["HISTORY_INSTITUTION"].getncattr("_FillValue"))))
        else:
            final_nc_data_prof["HISTORY_SOFTWARE"] = None
            final_nc_data_prof["HISTORY_SOFTWARE_RELEASE"] = None
            final_nc_data_prof["HISTORY_REFERENCE"] = None
            final_nc_data_prof["HISTORY_START_PRES"] = None
            final_nc_data_prof["HISTORY_STOP_PRES"] = None
            final_nc_data_prof["HISTORY_ACTION"] = None
            final_nc_data_prof["HISTORY_QCTEST"] = None
            final_nc_data_prof["HISTORY_PARAMETER"] = None
            final_nc_data_prof["HISTORY_STEP"] = None
            final_nc_data_prof["HISTORY_PREVIOUS_VALUE"] = None
            final_nc_data_prof["HISTORY_DATE"] = None
            final_nc_data_prof["HISTORY_INSTITUTION"] = None

        ##### SET FLAGS TO INDICATED WE'VE DONE DELAYED MODE PROCESSING #####
        # Set IP flag: we've operated on the complete input record 
        final_nc_data_prof = set_history_parems(final_nc_data_prof, "SET_IP")
        # Set ACTION to say we've performed tests
        final_nc_data_prof = set_history_parems(final_nc_data_prof, "SET_QCP$", HISTORY_QCTEST_names = ["Visual QC test"])
        
        ##### CHECK TO SEE IF WE'VE SET ANY BAD FLAGS FOR PARAMS ############
        parems_to_set = []

        if not np.array_equal(final_nc_data_prof["PRES_ADJUSTED_QC"], final_nc_data_prof["PRES_QC"]):
            parems_to_set.append("PRES")
        if not np.array_equal(final_nc_data_prof["PSAL_ADJUSTED_QC"], final_nc_data_prof["PSAL_QC"]):
            parems_to_set.append("PSAL")
        if not np.array_equal(final_nc_data_prof["TEMP_ADJUSTED_QC"], final_nc_data_prof["TEMP_QC"]):
            parems_to_set.append("TEMP")

        if parems_to_set:
            final_nc_data_prof = set_history_parems(final_nc_data_prof, "SET_QCF$", HISTORY_QCTEST_names=["Visual QC test"])
            # NOTE:
            # parems to set cannot be more than 3 parems
            final_nc_data_prof = set_history_parems(final_nc_data_prof, "SET_CF", parems_set=parems_to_set)

        ##### CHECK IF WE'VE INTERPOLATED POSITION/ TIME ####################
        if final_nc_data_prof["POSITION_QC"] == 8:
            # we've interp lat/lon
            final_nc_data_prof = set_history_parems(final_nc_data_prof, "SET_POS_INTERP")
        if final_nc_data_prof["JULD_QC"] == 8 or final_nc_data_prof["JULD_QC"] == 5:
            final_nc_data_prof = set_history_parems(final_nc_data_prof, "SET_JULD_INTERP")

        # Copy {PAREM}_ADJUSTED_QC arr vals into {PAREM}_QC
        final_nc_data_prof["PRES_QC"] = final_nc_data_prof["PRES_ADJUSTED_QC"]
        final_nc_data_prof["PSAL_QC"] = final_nc_data_prof["PSAL_ADJUSTED_QC"]
        final_nc_data_prof["TEMP_QC"] = final_nc_data_prof["TEMP_ADJUSTED_QC"]
        # Set CNDC QC vals to PSAL QC vals
        final_nc_data_prof["CNDC_QC"] = final_nc_data_prof["PSAL_ADJUSTED_QC"]
        final_nc_data_prof["CNDC_ADJUSTED_QC"] = final_nc_data_prof["PSAL_ADJUSTED_QC"]

        # 3.5.2 whereever PARAM_ADJUSTED_QC = 4, PARAM_ADJUSTED + PARAM_ADJUSTED_ERROR = FillVal!!
        final_nc_data_prof["TEMP_ADJUSTED"][np.where(final_nc_data_prof["TEMP_ADJUSTED_QC"] == 4)] = np.nan
        final_nc_data_prof["TEMP_ADJUSTED_ERROR"][np.where(final_nc_data_prof["TEMP_ADJUSTED_QC"] == 4)] = np.nan
        
        final_nc_data_prof["PRES_ADJUSTED"][np.where(final_nc_data_prof["PRES_ADJUSTED_QC"] == 4)] = np.nan
        final_nc_data_prof["PRES_ADJUSTED_ERROR"][np.where(final_nc_data_prof["PRES_ADJUSTED_QC"] == 4)] = np.nan
        
        final_nc_data_prof["PSAL_ADJUSTED"][np.where(final_nc_data_prof["PSAL_ADJUSTED_QC"] == 4)] = np.nan
        final_nc_data_prof["PSAL_ADJUSTED_ERROR"][np.where(final_nc_data_prof["PSAL_ADJUSTED_QC"] == 4)] = np.nan

        final_nc_data_prof["CNDC_ADJUSTED"][np.where(final_nc_data_prof["CNDC_ADJUSTED_QC"] == 4)] = np.nan
        final_nc_data_prof["CNDC_ADJUSTED_ERROR"][np.where(final_nc_data_prof["CNDC_ADJUSTED_QC"] == 4)] = np.nan

        make_final_nc_files(final_nc_data_prof, float_num, dest_filepath)

def main():

    float_num = "1902655"
    #float_num = "F10051"
    dest_filepath = "c:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\argo_to_nc\\Ascending\\F10051_final_A"
    nc_filepath = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\argo_to_nc\\Ascending\\F10051_FTR"
    orgargo_netcdf_filepath = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\RAW_DATA\\F10051_ARGO_NETCDF"
    config_fp = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\argo_to_nc\\ARGO_GEN\\F10051_final\\F10051_config_file.txt"
    """
    float_num = "F9186"
    dest_filepath = "c:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\csv_to_nc\\F9186_final"
    nc_filepath = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\csv_to_nc\\F9186_after_visual_inspection_new"
    config_fp = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\csv_to_nc\\F9186_final\\F9186_config_file.txt"
    orgargo_netcdf_filepath = None
    """
    if not os.path.exists(dest_filepath):
        os.mkdir(dest_filepath)

    """
    Pass in an ARGO NETCDF filepath to make config file for parems needed
    to make delayed mode NETCDF file.
    """
    #make_config_file(float_num, dest_filepath, org_argo_netcdf_filepath = orgargo_netcdf_filepath)
    #make_config_file(float_num, dest_filepath)

    process_data_dmode_files(nc_filepath, float_num, dest_filepath, config_fp, org_netcdf_fp = orgargo_netcdf_filepath)

if __name__ == '__main__':
    main()