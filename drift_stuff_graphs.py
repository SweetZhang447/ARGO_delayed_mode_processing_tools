from graphs_nc import del_bad_points
from tools import read_intermediate_nc_file
import numpy as np
import matplotlib.pyplot as plt
import gsw
from matplotlib.collections import LineCollection
import datetime
from matplotlib.dates import date2num, DateFormatter
import os
import glob
import mplcursors
import copy
from tools import read_intermediate_nc_file

def avg_var_graph(df_JULDs, avg_var_1, df_JULDs_2, avg_var_2, avg_var, float_name_1, float_name_2, difference=None, sign=None):
    
    # Plot both floats on the same graph
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_JULDs, avg_var_1, label=f'Float {float_name_1}', color='blue')
    
    if difference is not None and sign is not None:
        ax.plot(df_JULDs_2, avg_var_2, label=f'Float {float_name_2} {sign} {difference:.4f}', color='green')
    else:
        ax.plot(df_JULDs_2, avg_var_2, label=f'Float {float_name_2}', color='green')

    # Convert Julian times to datetime objects and set tick labels
    date_formatter = DateFormatter('%Y-%m-%d')
    ticks = [datetime.datetime(4713, 1, 1, 12) + datetime.timedelta(days=float(juld) - 1721425.5) for juld in ax.get_xticks()]
    ax.xaxis.set_major_formatter(date_formatter)
    ax.set_xticks(ax.get_xticks())  # Update tick positions
    ax.set_xticklabels([dt.strftime('%Y-%m-%d') for dt in ticks])

    ax.set_xlabel('Date')
    if avg_var == 'TEMP':
        ax.set_ylabel('Average Temperature (500-600 dbar)')
        ax.set_title(f'Average Temperature Comparison: Floats {float_name_1} and {float_name_2}')
    elif avg_var == 'PSAL':
        ax.set_ylabel('Average Salinity (500-600 dbar)')
        ax.set_title(f'Average Salinity Comparison: Floats {float_name_1} and {float_name_2}')
    else:
        raise Exception("avg_var must be either 'TEMP' or 'PSAL'")

    # Display legend and grid, and show plot
    ax.legend()
    ax.grid(True)
    plt.show()

# Function to average over depth
def average_depth_range(pres, var, depth_min, depth_max):
    # Create a boolean mask to filter the values in the desired depth range
    mask = (pres >= depth_min) & (pres <= depth_max)
    # Apply the mask to temperatures, setting invalid values to NaN to exclude from mean
    masked_temps = np.where(mask, var, np.nan)  # Keep original shape
    # Average the temperature over the valid depth range (along the depth axis)
    avg_temp = np.nanmean(masked_temps, axis=1)
    # avg_temp = np.nanmean(masked_temps)
    
    return avg_temp

def avg_var_graph_prof_num(df_JULDs, avg_var_1, df_JULDs_2, avg_var_2, avg_var, prof_num_1, prof_num_2, float_name_1, float_name_2, difference=None, sign=None):
    
    # Create dictionaries mapping each avg_var value to its profile number for both floats
    profile_dict_1 = {juld: profile for juld, profile in zip(df_JULDs, prof_num_1)}
    profile_dict_2 = {juld: profile for juld, profile in zip(df_JULDs_2, prof_num_2)}

    # Plot both floats on the same graph
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_JULDs, avg_var_1, label=f'Float {float_name_1}', color='blue')
    
    if difference is not None and sign is not None:
        ax.plot(df_JULDs_2, avg_var_2, label=f'Float {float_name_2} {sign} {difference:.4f}', color='green')
    else:
        ax.plot(df_JULDs_2, avg_var_2, label=f'Float {float_name_2}', color='green')

    plot_2 = ax.scatter(df_JULDs_2, avg_var_2, color='green', s=10)
    plot_1 = ax.scatter(df_JULDs, avg_var_1, color='blue', s=10)

    # Convert Julian times to datetime objects and set tick labels
    date_formatter = DateFormatter('%Y-%m-%d')
    ticks = [datetime.datetime(4713, 1, 1, 12) + datetime.timedelta(days=float(juld) - 1721425.5) for juld in ax.get_xticks()]
    ax.xaxis.set_major_formatter(date_formatter)
    ax.set_xticks(ax.get_xticks())  # Update tick positions
    ax.set_xticklabels([dt.strftime('%Y-%m-%d') for dt in ticks])

    ax.set_xlabel('Date')
    if avg_var == 'TEMP':
        ax.set_ylabel('Average Temperature (500-600 dbar)')
        ax.set_title(f'Average Temperature Comparison: Floats {float_name_1} and {float_name_2}')
    elif avg_var == 'PSAL':
        ax.set_ylabel('Average Salinity (500-600 dbar)')
        ax.set_title(f'Average Salinity Comparison: Floats {float_name_1} and {float_name_2}')
    else:
        raise Exception("avg_var must be either 'TEMP' or 'PSAL'")

    # Enable mplcursors hover functionality for each line plot
    cursor = mplcursors.cursor([plot_1, plot_2], hover=True)

    # Hover function using dictionaries
    def annotate_hover(sel):
        # Determine which line plot is being hovered and get profile number from dictionary
        line = sel.artist
        if line == plot_1:
            profile_number = profile_dict_1.get(sel.target[0], "N/A")
            sel.annotation.set_text(f"Float {float_name_1} Profile: {profile_number}")
        elif line == plot_2:
            profile_number = profile_dict_2.get(sel.target[0], "N/A")
            sel.annotation.set_text(f"Float {float_name_2} Profile: {profile_number}")
    
    cursor.connect("add", annotate_hover)

    # Display legend and grid, and show plot
    ax.legend()
    ax.grid(True)
    plt.show()

def main():
    nc_filepath = "c:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\F9186_nc_after_autochecks1"
    float_name_1 = "F9186"
    (PRESs, TEMPs, PSALs, COUNTs, 
     JULDs, JULD_LOCATIONs, LATs, LONs, JULD_QC, POSITION_QC, 
     PSAL_ADJUSTED, PSAL_ADJUSTED_ERROR, PSAL_ADJUSTED_QC, 
     TEMP_ADJUSTED, TEMP_ADJUSTED_ERROR, TEMP_ADJUSTED_QC, 
     PRES_ADJUSTED, PRES_ADJUSTED_ERROR, PRES_ADJUSTED_QC,
     PROFILE_NUMS, CNDC_ADJUSTED_QC) = read_intermediate_nc_file(nc_filepath)
    
    nc_filepath = "c:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\argo_to_nc\\F10051_1"
    float_name_2 = "F10051"
    (PRESs_2, TEMPs_2, PSALs_2, COUNTs_2, PTSCI_TIMES_2, 
     JULDs_2, LATs_2, LONs_2, JULD_QC_2, POSITION_QC_2, 
     PSAL_ADJUSTED_2, PSAL_ADJUSTED_ERROR_2, PSAL_ADJUSTED_QC_2, 
     TEMP_ADJUSTED_2, TEMP_ADJUSTED_ERROR_2, TEMP_ADJUSTED_QC_2, 
     PRES_ADJUSTED_2, PRES_ADJUSTED_ERROR_2, PRES_ADJUSTED_QC_2,
     PROFILE_NUMS_2) = read_intermediate_nc_file(nc_filepath)
    
    # Get rid of all nan slices
    # returns a bad_vals_mask in case further manipulation is needed w/ arr vals
    (PSAL_ADJUSTED, TEMP_ADJUSTED, PRES_ADJUSTED, 
     PROFILE_NUMS, LATs, LONs, JULDs, bad_vals_mask) = del_bad_points(PSAL_ADJUSTED, TEMP_ADJUSTED, PRES_ADJUSTED, 
                                                                      PROFILE_NUMS, LATs, LONs, JULDs)
    (PSAL_ADJUSTED_2, TEMP_ADJUSTED_2, PRES_ADJUSTED_2, 
     PROFILE_NUMS_2, LATs_2, LONs_2, JULDs_2, bad_vals_mask_2) = del_bad_points(PSAL_ADJUSTED_2, TEMP_ADJUSTED_2, PRES_ADJUSTED_2, 
                                                                                PROFILE_NUMS_2, LATs_2, LONs_2, JULDs_2)
    
    """
    # Find the index of where the dates overlap 
    index_1 = np.where((df_JULDs_2 >= df_JULDs[0]) & (df_JULDs_2 <= df_JULDs[-1]))
    # Gets data in same timeframe
    df_LATs_2 = df_LATs_2[index_1]
    df_LONs_2 = df_LONs_2[index_1]
    df_JULDs_2 = df_JULDs_2[index_1]
    df_PRESs_2 = df_PRESs_2[index_1]
    df_PSALs_2 = df_PSALs_2[index_1]
    df_TEMPs_2 = df_TEMPs_2[index_1]
    """

        # Average 500-600 depth for each day and plot 2 floats on same graphs
    # Depth range
    depth_min = 500
    depth_max = 700
    avg_var = 'PSAL'

    if avg_var == "TEMP":
        avg_temp_1 = average_depth_range(PRES_ADJUSTED, TEMP_ADJUSTED, depth_min, depth_max)
        avg_temp_2 = average_depth_range(PRES_ADJUSTED_2, TEMP_ADJUSTED_2, depth_min, depth_max)
        avg_var_graph(JULDs, avg_temp_1, JULDs_2, avg_temp_2, avg_var, float_name_1, float_name_2)
        avg_var_graph_prof_num(JULDs, avg_temp_1, JULDs_2, avg_temp_2, avg_var, PROFILE_NUMS, PROFILE_NUMS_2, float_name_1, float_name_2)
    
    if avg_var == "PSAL":
        avg_sal_1 = average_depth_range(PRES_ADJUSTED, PSAL_ADJUSTED, depth_min, depth_max)
        avg_sal_2 = average_depth_range(PRES_ADJUSTED_2, PSAL_ADJUSTED_2, depth_min, depth_max)

        # interpolate avg sal of ascent float 1 onto date range of descent float 2
        interpolated_sal = np.interp(JULDs_2, JULDs, avg_sal_1)
        # interpolate avg sal of float 2 onto date range of float 1
        interpolated_sal_2 = np.interp(JULDs, JULDs_2, avg_sal_2)
        difference = np.average(avg_sal_1 - interpolated_sal_2)
        sign = "+"

        # avg_var_graph(JULDs, avg_sal_1, JULDs, interpolated_sal_2 + difference, avg_var, float_name_1, float_name_2, difference, sign)
        # avg_var_graph(JULDs, avg_sal_1, JULDs_2, avg_sal_2, avg_var, float_name_1, float_name_2)
        
        # diff 
        """
        fig, ax = plt.subplots()
        # ax.plot(df_JULDs_2, interpolated_sal, label ="Interpolated Ascent onto Descent", color = 'green')
        #ax.plot(df_JULDs, avg_sal_1, label = "F10015 Ascent", color = 'blue')
        #ax.plot(df_JULDs_2, avg_sal_2, label = "F10015 Descent", color = 'red')
        ax.plot(JULDs_2, avg_sal_2 - interpolated_sal, label = "difference", color = "green")

        # Convert Julian times to datetime objects and set tick labels
        date_formatter = DateFormatter('%Y-%m-%d')
        ticks = [datetime.datetime(4713, 1, 1, 12) + datetime.timedelta(days=float(juld) - 1721425.5) for juld in ax.get_xticks()]
        ax.xaxis.set_major_formatter(date_formatter)
        ax.set_xticks(ax.get_xticks())  # Update tick positions
        ax.set_xticklabels([dt.strftime('%Y-%m-%d') for dt in ticks])
        ax.legend()
        ax.grid(True)
        ax.set_title("Difference Plots Ascent v Descent F10015 (Profiles 30-69)")
        plt.show()
        """
    
        # avg_var_graph_prof_num(JULDs, avg_sal_1, JULDs_2, avg_sal_2, avg_var, PROFILE_NUMS, PROFILE_NUMS_2, float_name_1, float_name_2)
    
if __name__ == '__main__':
    main()