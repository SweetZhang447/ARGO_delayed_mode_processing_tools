import argparse
import datetime
import glob
import os
import plotly as px
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import gsw
import netCDF4
import seaborn as sns
import matplotlib.dates as mdates
import gsw

def generate_TS(potential_temperature,conductivity,pressure):

    # Calculates practical salinity from conductivity
    prac_salinity = gsw.conversions.SP_from_C(conductivity,potential_temperature,pressure)
    # Calculate absolute salinity from practical salinity - coordinates are for disco bay
    salinity = gsw.conversions.SA_from_SP(prac_salinity,pressure, 69.165402, -51.265889)
    # Calculates conservative temp from potential temp
    temperature = gsw.conversions.CT_from_pt(salinity,potential_temperature)

    # removes outliers
    while salinity.min() < 28:
        salinity = np.delete(salinity, salinity.argmin())
        temperature = np.delete(temperature, temperature.argmin())
        pressure = np.delete(pressure, pressure.argmin())
    while salinity.max() > 35:
        salinity = np.delete(salinity, salinity.argmax())
        temperature = np.delete(temperature, temperature.argmax())
        pressure = np.delete(pressure, pressure.argmax())
    while temperature.min() < -2:
        salinity = np.delete(salinity, salinity.argmin())
        temperature = np.delete(temperature, temperature.argmin())
        pressure = np.delete(pressure, pressure.argmin())
    while temperature.max() > 5:
        salinity = np.delete(salinity, salinity.argmax())
        temperature = np.delete(temperature, temperature.argmax())
        pressure = np.delete(pressure, pressure.argmax())
    
    # gets bounds for graph
    smin = salinity.min() - 1
    smax = salinity.max() + 1
    tmin = temperature.min() - 1
    tmax = temperature.max() + 1

    # calculate number of gridcells needed in the x + y dimensions
    xdim = int(np.ceil((smax-smin)/0.1))
    ydim = int(np.ceil((tmax-tmin)))
    dens = np.zeros((ydim,xdim))
    # check API practical salinity + potenital temp 

    # Create temp and salt vectors of appropiate dimensions
    ti = np.linspace(0,ydim,ydim)+tmin
    si = np.linspace(1,xdim,xdim)*0.1+smin

    # Loop to fill in grid with densities
    for j in range(0,int(ydim)):
        for i in range(0, int(xdim)):    
            dens[j,i]=gsw.rho(si[i],ti[j],0)
 
    # Substract 1000 to convert to sigma-t
    dens = dens - 1000

    # Plot data
    plt.close('all')
    plt.figure(figsize=(10,6))
    CS = plt.contour(si,ti,dens, linestyles='dashed', colors='k')
    plt.clabel(CS, fontsize=12, inline=1, fmt='%.2f') # Label every second level
    plt.scatter(salinity, temperature, c=pressure,cmap=plt.cm.viridis, lw=0)
    plt.xlim([smin + .75, smax-0.75]); plt.ylim([tmin + 0.75, tmax - 0.75])
    plt.xlabel('Salinity (PSU)'); plt.ylabel('Temperature (degC)')
    plt.colorbar(label='Depth (m)')

    # adjust spacing, show plot
    plt.tight_layout()
    plt.show()

"""
This function reads a NetCDF file and converts them into numpy arrays.
"""
def read_netCDF(file_path):
   
    f =  netCDF4.Dataset(file_path)

    # Prints all keys of NetCDF file, depending on key name
    # set lists below
    # print(f.variables.keys())

    pressure = f.variables['PRES'][:].filled(np.NaN)
    temp = f.variables['TEMP'][:].filled(np.NaN)
    salinity = f.variables['PSAL'][:].filled(np.NaN)

    conductivity = gsw.C_from_SP(salinity, temp, pressure)

    return pressure, temp, salinity, conductivity

"""
This function reads a directory of NetCDF file and graphs a combined graph
"""
def read_netCDF_dir_combined_graph(file_path_dir):

    # Get all .nc files in directory
    nc_files_list = glob.glob(os.path.join(file_path_dir,'*.nc'))
    temperatures = []
    pressures = []
    for f in nc_files_list:
        file =  netCDF4.Dataset(f)
        # Prints all keys of NetCDF file
        # print(file.variables.keys())
        # Get needed param + convert to numpy arr
        temp = file.variables['TEMP'] 
        temp_np = temp[:]
        pressure = file.variables['PRES']
        pressure_np = pressure[:]
        # fill lists
        temperatures.append(temp_np)
        pressures.append(pressure_np)

    # combine array of different sizes, fill empty values with '0.'
    temperatures_stacked = stack_uneven_del(temperatures)
    pressure_stacked = stack_uneven_del(pressures)
    combined_array_temp = np.reshape(temperatures_stacked, (temperatures_stacked.shape[0], temperatures_stacked.shape[2]))
    combined_array_pressure = np.reshape(pressure_stacked, (pressure_stacked.shape[0], pressure_stacked.shape[2]))

    # graph combined arrays
    plt.close('all')
    plt.plot(combined_array_temp.T, combined_array_pressure.T, label = [1, 2, 3, 4, 5])
    plt.legend()
    plt.gca().invert_yaxis()
    plt.title("Pressure Versus Salinity for Float 6990591")
    plt.xlabel("Salinity")
    plt.ylabel("Pressure")
    plt.show()

def stack_uneven_fill(arrays, fill_value=0.):
    '''
    Takes numpy arrays of different sizes, fits them into a single numpy
    array (using the max size), fills extra values with `fill_value`.

    Args:
            arrays: list of np arrays of various sizes
                (must be same rank, but not necessarily same size)
            fill_value (float, optional):

    Returns:
            np.ndarray
    '''
    sizes = [a.shape for a in arrays]
    max_sizes = np.max(list(zip(*sizes)), -1)
    # The resultant array has stacked on the first dimension
    result = np.full((len(arrays),) + tuple(max_sizes), fill_value)
    for i, a in enumerate(arrays):
      # The shape of this array `a`, turned into slices
      slices = tuple(slice(0,s) for s in sizes[i])
      # Overwrite a block slice of `result` with this array `a`
      result[i][slices] = a
    return result

def stack_uneven_del(arrays):
    '''
    Takes a list of arrays, makes them all the same size by deleting extra 
    values, returns a single numpy array
    Args:
            arrays: list of np arrays of various sizes
                (must be same rank, but not necessarily same size)
            fill_value (float, optional):

    Returns:
            np.ndarray
    '''
    sizes = [a.shape for a in arrays]
    min_sizes = np.min(list(zip(*sizes)), -1)

    truncated_arrays = []
    for arr in arrays:
        # change shape of arr
        truncated_arr = arr[:len(arr), :min_sizes[1]]
        truncated_arrays.append(truncated_arr)
    result = np.stack(truncated_arrays)
    return result

"""
Generate scatterplot
"""
def generate_scatterplot(array1, array2):
    # Set seaborn graph theme
    sns.set_theme(style="whitegrid")
    # Make sure all plots are closed before execution
    plt.close('all')
    # We need to combine two arrays into one w/ pairs of values
    combined_array = np.vstack((array1, array2)).T
    # Generate scatterplot, X axis is first column, Y axis is second column
    sns.scatterplot(x=combined_array[:, 0], y=combined_array[:, 1])
    # Graph name, label...
    plt.title("Pressure Versus Temp Float 9186 Profile 4")
    plt.xlabel("Temp")
    plt.ylabel("Pressure")
    # Invert the Y-axis (if Y is pressure)
    plt.gca().invert_yaxis()
    plt.show()

def generate_lineplot(array1, array2):
    
    plt.close('all')
    # Combine the arrays into one with pairs of values
    combined_array = np.stack((array1, array2), axis=-1)

    # Create a Seaborn line plot
    sns.set(style="whitegrid")
    sns.lineplot(data=combined_array)

    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.title("Line Plot: Array1 vs. Array2")

    # Show the plot
    plt.show()

def read_netCDF_dir(file_path_dir):

    # Get all .nc files in directory
    nc_files_list = glob.glob(os.path.join(file_path_dir,'*.nc'))
    temperatures = []
    pressures = []
    dates = []
    org = datetime.datetime(1950, 1, 1)
    for f in nc_files_list:
        file =  netCDF4.Dataset(f)
        # Prints all keys of NetCDF file
        # print(file.variables.keys())
        # Get needed param + convert to numpy arr
        temp = file.variables['TEMP'] 
        temp_np = temp[:]
        pressure = file.variables['PRES']
        pressure_np = pressure[:]
        # Get date information
        julian_date = file.variables['JULD'][:]
        date = pd.to_datetime(julian_date,origin=org, unit='D')
        #date_t = np.array(date).flatten
        #date_char_to_string = date.strftime("%Y%m%d")
        # print(date_char_to_string)
        # fill lists
        temperatures.append(temp_np)
        pressures.append(pressure_np)

        # create list
        #temp = np.full(128,date)
        dates.append(date)

    plt.close('all')
    # combine array of different sizes, fill empty values with '0.'
    temperatures_stacked = stack_uneven_fill(temperatures)
    pressure_stacked = stack_uneven_fill(pressures)
    combined_array_temp = np.reshape(temperatures_stacked, (temperatures_stacked.shape[0], temperatures_stacked.shape[2]))
    combined_array_pressure = np.reshape(pressure_stacked, (pressure_stacked.shape[0], pressure_stacked.shape[2]))

    # Sample data
    dates = dates
    pressure = combined_array_pressure
    temperature = combined_array_temp

    # Set up the figure and axes
    fig, ax = plt.subplots()

    # Plotting each bar for pressure with temperature as color
    for i in range(len(dates)):
        bar = ax.bar(i, pressure[i], color=plt.cm.viridis((temperature[i] - np.min(temperature)) / (np.max(temperature) - np.min(temperature))))

    # Add labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Pressure')
    ax.set_title('Pressure over Time with Temperature Color')
    
    ax.invert_yaxis()
    # Adding colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=np.min(temperature), vmax=np.max(temperature)))
    sm.set_array([])  # You need to set a dummy array for the scalar mappable
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label('Temperature')

    # Set x-axis ticks and labels
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in dates], rotation=45, ha='right')  # Format and rotate date labels

    # Show the plot
    plt.show()


def main(file_path):
    
    # step 1:
    # Read in a NetCDF file, follow instructions inside of function, returns numpy array types
    pressure, temp, salinity, conductivity = read_netCDF(file_path)
    
    # Or, read in a CSV file
    # read_CSV(file_path)
    
    # Or, read a directory of NetCDF files and generate plots comparing all profiles
    # read_netCDF_dir_combined_graph(file_path)
    #read_netCDF_dir(file_path)
    
    # Generate scatterplot from any two arrays (X axis, Y axis)
    # Code to invert Y axis by default (pressure)
    # generate_scatterplot(temp, pressure)
    
    # Generate TS diagram
    generate_TS(temp,conductivity,pressure)


if __name__ == '__main__':

    """ user input snaz
    parser = argparse.ArgumentParser()
    parser.add_argument("-fp", "--nc_fp", action= "store",
                        help = "The file path of the NetCDF file.", dest= "nc_file_path",
                        type = str, required= True)
    
    args = parser.parse_args()
 
    nc_filepath = args.nc_file_path
    """
    #nc_filepath = 'C:\\Users\\szswe\\Desktop\\NETCDF'
    nc_filepath = '/home/sweet/Downloads/6990591_profiles/R6990591_005.nc'
    main(nc_filepath)