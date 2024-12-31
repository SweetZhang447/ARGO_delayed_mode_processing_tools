import csv
from datetime import datetime
import glob
import os
import netCDF4 as nc
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

def read_nc(filepath):

    dataset = nc.Dataset(filepath)

    # Get vars
    JULD = dataset.variables['JULD'][:]
    lat = dataset.variables['LATITUDE'][:]
    lon = dataset.variables['LONGITUDE'][:]

    # pressure
    pressure = dataset.variables['PRES'][:]
    salinity = dataset.variables['PSAL'][:]
    temp = dataset.variables['TEMP'][:]

    cyc_num = dataset.variables['CYCLE_NUMBER'][:]
    cyc_num_25 = np.where(cyc_num == 25)[0]

    print(cyc_num_25.shape)

    return JULD, lat, lon, pressure, salinity, temp
def var_time(data, data_type, float_name, profilenum, dest_dir):
    """
    Makes a scatter plot for a variable defined by data_flag against time.
    data_type:
        1 - Pressure
        2 - Temperature,
        3 - Salinity,
        4 - Conductivity, 
        5 - I
    """

    timestamps = []
    arr1 = []
    ylabel = "null"

    for row in data:
        date = datetime.strptime(row[1], '%Y%m%dT%H%M%S')
        timestamps.append(date)
        if data_type == 1:
            arr1.append(row[2])
            ylabel = "Pressure"
        elif data_type == 2:
            arr1.append(row[3])
            ylabel = "Temperature"
        elif data_type == 3:
            arr1.append(row[4])
            ylabel = "Salinity"
        elif data_type == 4:
            arr1.append(row[5])
            ylabel = "Conductivity"
        elif data_type == 5:
            arr1.append(row[6])
            ylabel = "I" 
        else:
            print("Invalid data_type flag: {}, flag must be 1-4".format(data_type))
            return
    
    arr1_np = np.asarray(arr1, dtype=np.float64)

    timestamps = np.arange(1, len(arr1_np) + 1)

    fig, ax = plt.subplots()
    ax.scatter(timestamps, arr1_np)
    # Set date labels

    #ax.set_ylabel(np.arange(int(max), int(min), 100))
    #plt.gca().invert_yaxis()

    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title("{}-Time for {} dive {} on {}".format(ylabel, float_name, profilenum, date.date()))
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.grid()

    plt.show()

def read_sci(file_path):

    list_of_csvs = glob.glob(os.path.join(file_path,'*.science_log.csv')) 

    ALL_LGR_P = []
    ALL_LGR_PTSCI = []
    ALL_LGR_CP_PTSCI = []
    ALL_LGR_combined = []
    EXCLUDE_CP_TIMESTAMPS = []
    
    for file in list_of_csvs:

        # Open CSV file + read in values 
        temp_LGR_P = []
        temp_LGR_PTSCI = []
        temp_LGR_CP_PTSCI = []
        temp_LGR_combined = []
        counter = 0

        with open(file, newline ='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter = ',')
            rows = 0
            for row in csv_reader:
                
                # Contains Pressure + Time values for decent + ascent
                if row[0] =='LGR_P':
                    temp_LGR_P.append(row)
                    temp_LGR_combined.append(row)
                # Contains info PTSCI for bottom + ascent
                elif row[0] == 'LGR_PTSCI':
                    temp_LGR_PTSCI.append(row)
                    temp_LGR_combined.append(row)
                    if counter == 25:
                        rows = rows + 1
                # Contains profile info
                elif row[0] == 'LGR_CP_PTSCI':
                    temp_LGR_CP_PTSCI.append(row)
        counter = counter + 1
        ALL_LGR_P.append(temp_LGR_P)
        ALL_LGR_PTSCI.append(temp_LGR_PTSCI)
        ALL_LGR_CP_PTSCI.append(temp_LGR_CP_PTSCI)
        ALL_LGR_combined.append(temp_LGR_combined)
       

        start_date = datetime.strptime(temp_LGR_CP_PTSCI[0][1], '%Y%m%dT%H%M%S')
        end_date =  datetime.strptime(temp_LGR_CP_PTSCI[len(temp_LGR_CP_PTSCI) - 1][1], '%Y%m%dT%H%M%S')
        """
        for val in temp_LGR_P:
            val_date = datetime.strptime(val[1], '%Y%m%dT%H%M%S')
            if not start_date <= val_date <= end_date:
                EXCLUDE_CP_TIMESTAMPS.append(val)
        """
        for val in temp_LGR_PTSCI:
            val_date = datetime.strptime(val[1], '%Y%m%dT%H%M%S')
            if not start_date <= val_date <= end_date:
                EXCLUDE_CP_TIMESTAMPS.append(val)
    print(rows)
    return ALL_LGR_P, ALL_LGR_PTSCI, ALL_LGR_CP_PTSCI, ALL_LGR_combined, EXCLUDE_CP_TIMESTAMPS

def main():

    # Directory of less profile float - science logs
    sci_path = "C:\\Users\\szswe\\Downloads\\F10051_all_data"
    ALL_LGR_P, ALL_LGR_PTSCI, ALL_LGR_CP_PTSCI, ALL_LGR_combined, EXCLUDE_CP_TIMESTAMPS = read_sci(sci_path)
    print(len(EXCLUDE_CP_TIMESTAMPS))
    #var_time(EXCLUDE_CP_TIMESTAMPS, 1, "test", "dawg", None)

    # NetCDF
    nc_path = "C:\\Users\\szswe\\Desktop\\NETCDF\\1902655_Rtraj.nc"
    JULD, lat, lon, pressure, salinity, temp = read_nc(nc_path)
    print(pressure)
    """
    counter = 0
    for val in pressure:
        if isinstance(val,np.float32):
            counter = counter + 1
    print(counter)
    """
if __name__ == '__main__':

    """ user input snaz
    parser = argparse.ArgumentParser()
    parser.add_argument("-fp", "--nc_fp", action= "store",
                        help = "The file path of the NetCDF file.", dest= "nc_file_path",
                        type = str, required= True)
    
    args = parser.parse_args()
 
    nc_filepath = args.nc_file_path
    """
    # filepath = 'C:\\Users\\szswe\\Desktop\\NETCDF'
    # nc_filepath = 'C:\\Users\\szswe\\Desktop\\FLOAT_LOAD\\F9186\\Processed\\004\\data_snapshots\\F9186-004.nc'
    main()