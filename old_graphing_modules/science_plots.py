import argparse
import csv
import glob
import os
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime

def speed(data, float_name, profilenum, dest_dir):

    timestamps = []
    pressure = []

    for row in data:
        date = datetime.strptime(row[1], '%Y%m%dT%H%M%S')
        timestamps.append(date) 
        pressure.append(float(row[2]))

    time_elapsed = []
    adjacent_pressures = []

    for i in range(1, len(timestamps)):
        
        # Get the amount of time passed
        prevday = timestamps[i - 1]
        theday = timestamps[i]
        temp_sub = theday - prevday
        # Get adjacent pressures
        prevPressure = pressure[i - 1]
        thePressure = pressure[i]
        temp_pressure = thePressure - prevPressure
        time_elapsed.append(temp_sub.total_seconds())
        adjacent_pressures.append(temp_pressure)
    
    time_elapsed_np = np.asarray(time_elapsed, dtype=np.float64)
    pressure_np = np.asarray(adjacent_pressures, dtype=np.float64)
    speed_np = np.divide(pressure_np, time_elapsed_np)
    # Make timestamps and speed same length
    timestamps.pop()

    fig, ax = plt.subplots()
    ax.scatter(timestamps, speed_np)
    # Set date labels
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.xlabel("Time")
    plt.ylabel("Speed (Pressure/ Second)")
    plt.title("Speed for {} dive {} on {}".format(float_name, profilenum, date.date()))
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.grid()
    if dest_dir == None:
        plt.show()
    else:
        plt.savefig(os.path.join(dest_dir,"Speed_{}_{}.png".format(float_name, profilenum)))

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

    fig, ax = plt.subplots()
    ax.scatter(timestamps, arr1_np)
    # Set date labels
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    max = np.nanmax(arr1_np)
    min = np.nanmin(arr1_np)

    if data_type == 1:
        ax.set_ylabel(np.arange(int(max), int(min), 100))
        plt.gca().invert_yaxis()

    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title("{}-Time for {} dive {} on {}".format(ylabel, float_name, profilenum, date.date()))
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.grid()
    if dest_dir == None:
        plt.show()
    else:
        plt.savefig(os.path.join(dest_dir, "{}-Time_{}_{}.png".format(ylabel, float_name, profilenum)))

def single_profile_info(file_path):
    """
    Read CSV file, returns lists for LGR_P, LGR_PTSCI, LGR_CP_PTSCI
    """
    # Open CSV file + read in values 
    LGR_P = []
    LGR_PTSCI = []
    LGR_CP_PTSCI = []
    LGR_combined = []
    LGR_PTSCI_DESCENT = []
    LGR_PTSCI_ASCENT = []

    descent = False
    ascent = False
    with open(file_path, newline ='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter = ',')
        for row in csv_reader:
            ########## get info for descent
            if row[2] == 'Park Descent Mission':
                descent = True
            if row[2] == 'Park Mission':
                descent = False
            if descent == True:
                if row[0] == 'LGR_PTSCI':
                    LGR_PTSCI_DESCENT.append(row)
            ######## get info for ascent
            if row[2] == 'ASCENT':
                ascent = True
            if row[2] == 'CP already finished':
                ascent = False
            if ascent == True:
                if row[0] == 'LGR_PTSCI':
                    LGR_PTSCI_ASCENT.append(row)

            # Contains Pressure + Time values for decent + ascent
            if row[0] =='LGR_P':
                LGR_P.append(row)
                LGR_combined.append(row)
            # Contains info PTSCI for bottom + ascent
            elif row[0] == 'LGR_PTSCI':
                LGR_PTSCI.append(row)
                LGR_combined.append(row)
            # Contains profile info
            elif row[0] == 'LGR_CP_PTSCI':
                LGR_CP_PTSCI.append(row)
                

    return LGR_P, LGR_PTSCI, LGR_CP_PTSCI, LGR_combined, LGR_PTSCI_DESCENT, LGR_PTSCI_ASCENT

def var_pressure(data_list, data_type, float_name, profilenum, dest_dir):
    """
    Makes a scatter plot for a variable defined by data_flag against pressure.
    data_type:
        1 - Temperature,
        2 - Salinity,
        3 - Conductivity, 
        4 - I
    """ 
    # Pressure versus Salinity
    arr1 = []
    pressures = []
    xlabel = "null"

    for row in data_list:
        if data_type == 1:
            arr1.append(row[3])
            xlabel = "Temperature"
        elif data_type == 2:
            arr1.append(row[4])
            xlabel = "Salinity"
        elif data_type == 3:
            arr1.append(row[5])
            xlabel = "Conductivity"
        elif data_type == 4:
            arr1.append(row[6])
            xlabel = "I"
        else:
            print("Invalid data_type flag: {}, flag must be 1-4".format(data_type))
            return
        
        pressures.append(row[2])

    pressures_np = np.asarray(pressures, dtype=np.float64)
    arr1_np = np.asarray(arr1, dtype=np.float64)

    fig, ax = plt.subplots()
    ax.scatter(arr1_np, pressures_np)

    # Set Y-axis ticks/ scale
    ymax = np.max(pressures_np)
    ymin = np.min(pressures_np)
    ax.set_ylabel(np.arange(int(ymax), int(ymin), 100))

    plt.xlabel(xlabel)
    plt.ylabel("Pressure")
    plt.title("{}-Pressure for {} dive {}".format(xlabel, float_name, profilenum))
    plt.gca().invert_yaxis()
    plt.grid()
    if dest_dir == None:
        plt.show()
    else:
        plt.savefig(os.path.join(dest_dir, "{}-Pressure_{}_{}.png".format(xlabel, float_name, profilenum)))

def directory_sci_info(file_path, lower_range, higher_range, splitname):
    
    list_of_csvs = glob.glob(os.path.join(file_path,'*.science_log.csv')) 

    ALL_LGR_P = []
    ALL_LGR_PTSCI = []
    ALL_LGR_CP_PTSCI = []
    ALL_LGR_combined = []
    profile_nums = []

    if lower_range == 0 and higher_range == 0:
 
        for file in list_of_csvs:

            basename = os.path.basename(file).split(splitname)
            num_basename = int(basename[1].split(".")[0])
            profile_nums.append(num_basename)
            
            # Open CSV file + read in values 
            temp_LGR_P = []
            temp_LGR_PTSCI = []
            temp_LGR_CP_PTSCI = []
            temp_LGR_combined = []

            with open(file, newline ='') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter = ',')
                PTSCI_descent_flag = False
                for row in csv_reader:
                    # Contains Pressure + Time values for decent + ascent
                    if row[0] =='LGR_P':
                        temp_LGR_P.append(row)
                        temp_LGR_combined.append(row)
                    # Contains info PTSCI for bottom + ascent
                    elif row[0] == 'LGR_PTSCI':
                        temp_LGR_PTSCI.append(row)
                        temp_LGR_combined.append(row)
                    # Contains profile info 
                    elif row[0] == 'LGR_CP_PTSCI':
                        temp_LGR_CP_PTSCI.append(row)
                    """
                    if(row[2] == 'Park Descent Mission'):
                        PTSCI_descent_flag = True
                    if(row[2] == 'Park Mission'):
                        PTSCI_descent_flag = False
                    if PTSCI_descent_flag == True:
                        if row[0] == 'LGR_PTSCI':
                            temp_LGR_CP_PTSCI.append(row)
                    """
                    
                        
            ALL_LGR_P.append(temp_LGR_P)
            ALL_LGR_PTSCI.append(temp_LGR_PTSCI)
            ALL_LGR_CP_PTSCI.append(temp_LGR_CP_PTSCI)
            ALL_LGR_combined.append(temp_LGR_combined)

        return ALL_LGR_P, ALL_LGR_PTSCI, ALL_LGR_CP_PTSCI, ALL_LGR_combined, profile_nums
    
    else:

        within_range = False
        for file in list_of_csvs:

            basename = os.path.basename(file).split(splitname)
            num_basename = int(basename[1].split(".")[0])

            if num_basename == lower_range:
                within_range = True
             
            if within_range == True:

                profile_nums.append(num_basename)

                # Open CSV file + read in values 
                temp_LGR_P = []
                temp_LGR_PTSCI = []
                temp_LGR_CP_PTSCI = []
                temp_LGR_combined = []
                
                with open(file, newline ='') as csvfile:
                    csv_reader = csv.reader(csvfile, delimiter = ',')
                    for row in csv_reader:
                        # Contains Pressure + Time values for decent + ascent
                        if row[0] =='LGR_P':
                            temp_LGR_P.append(row)
                            temp_LGR_combined.append(row)
                        # Contains info PTSCI for bottom + ascent
                        elif row[0] == 'LGR_PTSCI':
                            temp_LGR_PTSCI.append(row)
                            temp_LGR_combined.append(row)
                        # Contains profile info
                        elif row[0] == 'LGR_CP_PTSCI':
                            temp_LGR_CP_PTSCI.append(row)
                            
                ALL_LGR_P.append(temp_LGR_P)
                ALL_LGR_PTSCI.append(temp_LGR_PTSCI)
                ALL_LGR_CP_PTSCI.append(temp_LGR_CP_PTSCI)
                ALL_LGR_combined.append(temp_LGR_combined)
            
            if within_range == True and num_basename == higher_range:
                within_range = False

        return ALL_LGR_P, ALL_LGR_PTSCI, ALL_LGR_CP_PTSCI, ALL_LGR_combined, profile_nums

def all_var_time(data, data_type, float_name, profile_num, dest_dir):
    """
    Makes a scatter plot for a variable defined by data_flag against time for
    all profiles in a directory.
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
     
    for profile in data:
        temp_timestamps = []
        temp = []
        for row in profile:
            date = datetime.strptime(row[1], '%Y%m%dT%H%M%S')
            temp_timestamps.append(date)
            if data_type == 1:
                temp.append(row[2])
                ylabel = "Pressure"
            elif data_type == 2:
                temp.append(row[3])
                ylabel = "Temperature"
            elif data_type == 3:
                temp.append(row[4])
                ylabel = "Salinity"
            elif data_type == 4:
                temp.append(row[5])
                ylabel = "Conductivity"
            elif data_type == 5:
                temp.append(row[6])
                ylabel = "I" 
            else:
                print("Invalid data_type flag: {}, flag must be 1-4".format(data_type))
                return
            
        arr1.append(temp)
        timestamps.append(temp_timestamps)

    fig, ax = plt.subplots()
    for i in range(len(arr1)):
        arr1_np = np.asarray(arr1[i], dtype=np.float64)
        timestamps_np = np.asarray(timestamps[i], dtype = 'datetime64')
        # Plot Scatter Graph
        ax.scatter(timestamps_np, arr1_np, label=f'Dive {profile_num[i]}') 
    
    
    if data_type == 1:    
        max = np.max(arr1_np)
        min = np.min(arr1_np)
        ax.set_ylabel(np.arange(int(max), int(min), 100))
        plt.gca().invert_yaxis()

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title("{} Versus Time for {} Dives {} - {}".format(ylabel, float_name, profile_num[0], profile_num[len(profile_num)- 1]))
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.grid()
    if dest_dir == None:
        plt.show()
    else:
        plt.savefig(os.path.join(dest_dir,"{}-Time_{}_{}-{}.png".format(ylabel, float_name, profile_num[0], profile_num[len(profile_num)- 1])))

def all_var_pressure(data, data_type, float_name, profilenum, dest_dir):
    """
    Makes a scatter plot for a variable defined by data_flag against pressure for
    all profiles in a directory
    data_type:
        1 - Temperature,
        2 - Salinity,
        3 - Conductivity, 
        4 - I
    """ 

    arr1 = []
    pressures = []
    xlabel = "null"

    for profile in data:
        temp = []
        temp_pressures = []
        for row in profile:
            temp_pressures.append(row[2])
            if data_type == 1:
                temp.append(row[3])
                xlabel = "Temperature"
            elif data_type == 2:
                temp.append(row[4])
                xlabel = "Salinity"
            elif data_type == 3:
                temp.append(row[5])
                xlabel = "Conductivity"
            elif data_type == 4:
                temp.append(row[6])
                xlabel = "I"
            else:
                print("Invalid data_type flag: {}, flag must be 1-4".format(data_type))
                return
            
        arr1.append(temp)
        pressures.append(temp_pressures)

    fig, ax = plt.subplots()
    for i in range(len(arr1)):
        arr1_np = np.asarray(arr1[i], dtype=np.float64)
        pressures_np = np.asarray(pressures[i], dtype=np.float64)
        # Plot Scatter Graph
        ax.scatter(arr1_np, pressures_np, label=f'Dive {profilenum[i]}') 
    
    print(arr1_np)
    max = np.nanmax(arr1_np)
    min = np.nanmin(arr1_np)
    ax.set_ylabel(np.arange(int(max), int(min), 100))
    plt.gca().invert_yaxis()

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlabel(xlabel)
    plt.ylabel("Pressure")
    plt.title("Pressure Versus {} for {} Dives {} - {}".format(xlabel, float_name, profilenum[0], profilenum[len(profilenum) - 1]))
    plt.grid()
    if dest_dir == None:
        plt.show()
    else:
        plt.savefig(os.path.join(dest_dir,"Pressure-{}_{}_{}-{}".format(xlabel, float_name, profilenum[0], profilenum[len(profilenum) - 1])))

def all_speed(data, float_name, profilenum, dest_dir):

    timestamps = []
    pressures = []
     
    for profile in data:
        temp_timestamps = []
        temp_pressures = []
        for row in profile:
            date = datetime.strptime(row[1], '%Y%m%dT%H%M%S')
            temp_timestamps.append(date)  
            temp_pressures.append(float(row[2]))
        pressures.append(temp_pressures)
        timestamps.append(temp_timestamps)
    
    # Calculate a list of times elapsed for all profiles
    time_elapsed = []
    adjacent_pressures = []
    for i in range(len(pressures)):
        time_elapsed_temp = [] 
        adjacent_pressures_temp = []
        for j in range(1, len(timestamps[i])):
            prevday = timestamps[i][j - 1]
            theday = timestamps[i][j]
            temp_sub = theday - prevday
            time_elapsed_temp.append(temp_sub.total_seconds())
            # Get adjacent pressures
            prevPressure = pressures[i][j - 1]
            thePressure = pressures[i][j]
            temp_pressure = thePressure - prevPressure
            adjacent_pressures_temp.append(temp_pressure)
        time_elapsed.append(time_elapsed_temp)
        adjacent_pressures.append(adjacent_pressures_temp)

    fig, ax = plt.subplots()
    for i in range(len(pressures)):
        pressures_np = np.asarray(adjacent_pressures[i], dtype=np.float64)
        time_elapsed_np = np.asarray(time_elapsed[i], dtype=np.float64)
        speed_np = np.divide(pressures_np, time_elapsed_np)      
        # Plot Scatter Graph
        timestamps[i].pop()
        ax.scatter(timestamps[i], speed_np, label=f'Dive {profilenum[i]}') 

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlabel("Time")
    plt.ylabel("Speed")
    plt.title("Speed for {} Dives {} - {}".format(float_name, profilenum[0], profilenum[len(profilenum)-1]))
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.grid()
    if dest_dir == None:
        plt.show()
    else:
        plt.savefig(os.path.join(dest_dir, "Speed_{}_{}-{}".format(float_name, profilenum[0], profilenum[len(profilenum)-1])))

def all_pressure_var_time(data, float_name, data_flag, profilenum, dest_dir):
    """
    Makes scatter plots for a variable defined by data_flag against the date and
    the pressure
        1 - Temperature,
        2 - Salinity,
        3 - Conductivity, 
        4 - I
    """

    timestamps = []
    pressures = []
    var = []
    var_label = "null"
     
    for profile in data:
        temp_timestamps = []
        temp_pressures = []
        temp_var = []
        for row in profile:
            date = datetime.strptime(row[1], '%Y%m%dT%H%M%S')
            temp_timestamps.append(date) 
            temp_pressures.append(row[2])
            if data_flag == 1:
                var_label = "Temperature"
                temp_var.append(row[3])
            elif data_flag == 2:
                var_label = "Salinity"
                temp_var.append(row[4])
            elif data_flag == 3:
                var_label = "Conductivity"
                temp_var.append(row[5])
            elif data_flag == 4:
                var_label = "I"
                temp_var.append(row[6])     
            else:
                print("Invalid data_type flag: {}, flag must be 1-4".format(data_flag))
                return                     
        pressures.append(temp_pressures)
        timestamps.append(temp_timestamps)
        var.append(temp_var)
    
    fig, ax = plt.subplots()

    for i in range(len(pressures)):
        pressures_np = np.asarray(pressures[i], dtype=np.float64)
        timestamps_np = np.asarray(timestamps[i], dtype = 'datetime64[D]')
        var_np = np.asarray(var[i], dtype=np.float64)
        
        # Plot Scatter Graph
        # JOSH: vmin=-2, vmax=5 should we set 
        scatter = ax.scatter(timestamps_np, pressures_np, c=var_np, cmap='jet', marker='s', s=100) 

    # Add colorbar to show temperature values
    cbar = plt.colorbar(scatter)
    cbar.set_label(var_label)
    
    plt.gca().invert_yaxis()
    plt.xlabel("Date")
    plt.ylabel("Pressure")
    plt.title("Argo Float {} Dives {} - {}".format(float_name, profilenum[0], profilenum[len(profilenum) - 1]))
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.grid()
    if dest_dir == None:
        plt.show()
    else:
        plt.savefig(os.path.join(dest_dir, "Date-Pressure-{}_{}_{}-{}".format(var_label, float_name, profilenum[0], profilenum[len(profilenum) - 1])))
 
def main(nc_filepath, single_or_double, float_name, lower_range, higher_range, dest_dir):

    plt.close('all')
    # put in the part of the filename before the profile number
    splitname = "032-4822."
    
    if single_or_double == 0:

        profilenum =  os.path.basename(nc_filepath).split(splitname)[1].split(".")[0]
        
        # Get data info
        nc_filepath = "C:\\Users\\szswe\\Desktop\\F10051_processed\\030\\032-4822.030.20231202T214338.science_log.csv"
        nc_filepath = "C:\\Users\\szswe\\Desktop\\F10051_processed\\029\\032-4822.029.20231127T215758.science_log.csv"
        LGR_P, LGR_PTSCI, LGR_CP_PTSCI, LGR_combined, LGR_PTSCI_DESCENT, LGR_PTSCI_ASCENT = single_profile_info(nc_filepath)
        # NOTE: LGR_combined is LGR_P and LGR_PTSCI

        # Make scatter plots for various variables versus time
        # Variables defined by data_flag 1-5
        # var_time(LGR_combined, 1, float_name, profilenum, dest_dir)
        var_time(LGR_CP_PTSCI, 1, float_name, profilenum, dest_dir)
        var_time(LGR_CP_PTSCI, 2, float_name, profilenum, dest_dir)
        var_time(LGR_CP_PTSCI, 3, float_name, profilenum, dest_dir)

        # Make scatter plot for various variables versus pressure
        # Variables defined by data_flag 1-4
        var_pressure(LGR_CP_PTSCI, 1, float_name, profilenum, dest_dir)
        var_pressure(LGR_CP_PTSCI, 2, float_name, profilenum, dest_dir)

        # Make VROOOM speed graph
        # speed(LGR_P, float_name, profilenum, dest_dir)

    if single_or_double == 1:

        # Get data info
        ALL_LGR_P, ALL_LGR_PTSCI, ALL_LGR_CP_PTSCI, ALL_LGR_combined, profilenum = directory_sci_info(nc_filepath, lower_range, higher_range, splitname)
        # NOTE: LGR_combined is LGR_P and LGR_PTSCI
        all_var_time(ALL_LGR_CP_PTSCI, 3, float_name, profilenum, dest_dir)
        all_var_pressure(ALL_LGR_CP_PTSCI, 2, float_name, profilenum, dest_dir)
        # all_speed(ALL_LGR_PTSCI, float_name, profilenum, dest_dir)
        # all_pressure_var_time(ALL_LGR_CP_PTSCI, float_name, 3, profilenum, dest_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--nc_fp", action= "store",
                        help = "The file path of the CSV file.", dest= "nc_file_path",
                        type = str, required= True)
    
    parser.add_argument("-d", "--dest_dir", action= "store",
            help = "Don't pass in to show, pass in path to save.", dest= "dest_dir",
            type = str, required = False)
    
    parser.add_argument("-s", "--single_or_dir", action= "store",
                    help = "Pass in 0 if generating graphs for one profile, 1 if directory of profiles", dest= "single_or_dir",
                    type = int, required= True) 
    
    parser.add_argument("-fn", "--float_name", action= "store",
                help = "Name of Float to look at", dest= "float_name",
                type = str, required= True)
    
    parser.add_argument("-r", "--range_of_dives", action= "store",
                help = "Pass in a range of profile numbers to generate graph for (EX. 4-7) or 0-0 to look at all", dest= "range_of_dives",
                type = str, required = False)
    
    
    args = parser.parse_args()
 
    nc_filepath = args.nc_file_path
    single_or_double = args.single_or_dir
    float_name = args.float_name
    range_of_dives = args.range_of_dives
    dest_dir = args.dest_dir
    
    # Error checking user input
    if single_or_double == 1 and range_of_dives == None:
        raise Exception("Must specify range of dives numbers to look at!")

    # Make sure range_of_dives is valid
    if single_or_double == 1 and range_of_dives != None:
        range_temp = range_of_dives.split("-")
        try:
            lower_range = int(range_temp[0])
            higher_range = int(range_temp[1])
            if lower_range > higher_range:
                raise Exception("Please specify valid range, {} is more than {}!".format(lower_range, higher_range))
        except Exception as e:
            print("Invalid user input")
            print(e)
        
        main(nc_filepath, single_or_double, float_name, lower_range, higher_range, dest_dir)

    if single_or_double == 0:
        lower_range = 0
        higher_range = 0
        main(nc_filepath, single_or_double, float_name, lower_range, higher_range, dest_dir)
    
    # single
    # nc_filepath = 'C:\\Users\\szswe\\Desktop\\FLOAT_LOAD\\F9186\\Processed\\004\\008-1520.004.20200916T200702.science_log.csv'
    # dir
    # nc_filepath = "C:\\Users\\szswe\\Desktop\\science_logs"
    # alotta them: "C:\Users\szswe\Downloads\F9186\F9186\Processed"
    