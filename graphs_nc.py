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
from tools import from_julian_day, read_intermediate_nc_file, to_julian_day
from matplotlib.lines import Line2D

def TS_graph_double(df_SALs, df_TEMPs, df_JULD, df_LONs, df_LATs, df_PRESs, 
        df_LATs_2, df_LONs_2, df_JULDs_2, df_PRESs_2, df_PSALs_2, df_TEMPs_2,
        float_name_1, float_name_2):
    
    # convert data for F9443
    for i in np.arange(df_SALs.shape[0]):    # number of profiles
        df_SALs[i, :] = gsw.conversions.SA_from_SP(df_SALs[i, :], df_TEMPs[i, :], df_LONs[i], df_LATs[i])
        df_TEMPs[i, :] = gsw.conversions.CT_from_t(df_SALs[i, :], df_TEMPs[i, :], df_PRESs[i, :])

    # convert data for F9186
    for i in np.arange(df_PSALs_2.shape[0]):    # number of profiles
        df_PSALs_2[i, :] = gsw.conversions.SA_from_SP(df_PSALs_2[i, :], df_TEMPs_2[i, :], df_LONs_2[i], df_LATs_2[i])
        df_TEMPs_2[i, :] = gsw.conversions.CT_from_t(df_PSALs_2[i, :], df_TEMPs_2[i, :], df_PRESs_2[i, :])

    # Define salinity and temperature bounds for the contour plot based on ref dataset
    smin = np.nanmin(df_SALs) - 1
    smax = np.nanmax(df_SALs) + 1
    tmin = np.nanmin(df_TEMPs) - 1
    tmax = np.nanmax(df_TEMPs) + 1

    # Calculate number of grid cells needed in the x and y dimensions
    xdim = int(np.ceil((smax - smin) / 0.1))
    ydim = int(np.ceil((tmax - tmin)))
    dens = np.zeros((ydim, xdim))

    # Create temp and salt vectors of appropriate dimensions
    ti = np.linspace(0, ydim, ydim) + tmin
    si = np.linspace(1, xdim, xdim) * 0.1 + smin

    # Loop to fill in grid with densities
    for j in range(ydim):
        for i in range(xdim):
            dens[j, i] = gsw.rho(si[i], ti[j], 0)

    # Subtract 1000 to convert to sigma-t
    dens = dens - 1000
    
    # Prepare F9443 dataset points
    ref_segments = np.stack((df_SALs, df_TEMPs), axis=2)
    # Create a LineCollection for F9443
    ref_lc = LineCollection(ref_segments, array = df_JULD, cmap='Purples', alpha=0.7)
    ref_lc.set_linewidth(1)

    # Prepare F9186 dataset points
    F9186_segments = np.stack((df_PSALs_2, df_TEMPs_2), axis=2)
    # Create a LineCollection for F9186
    F9186_lc = LineCollection(F9186_segments, array = df_JULDs_2, cmap='Reds', alpha=0.5)
    F9186_lc.set_linewidth(1)

    # Plot data
    fig, ax = plt.subplots(figsize=(10,6))
    CS = plt.contour(si, ti, dens, linestyles='dashed', colors='k')
    plt.clabel(CS, fontsize=12, inline=1, fmt='%.2f')

    # Add LineCollections to the plot
    ax = plt.gca()
    ax.add_collection(ref_lc)
    ax.add_collection(F9186_lc)

    # Add colorbar with date formatter
    cbar = plt.colorbar(ref_lc, ax=ax, label='Date')
    date_formatter = DateFormatter('%Y-%m-%d')

    # Convert colorbar ticks to regular dates
    cbar_ticks = [datetime.datetime(1950, 1, 1) + (datetime.timedelta(days=float(juld))) for juld in cbar.get_ticks()]
    cbar.ax.yaxis.set_major_formatter(date_formatter)
    cbar.set_ticklabels([dt.strftime('%Y-%m-%d') for dt in cbar_ticks])

    plt.xlim([smin + 0.75, smax - 0.75])
    plt.ylim([tmin + 0.75, tmax - 0.75])
    plt.xlabel('Salinity (PSU)')
    plt.ylabel('Temperature (degC)')
    plt.title(f"Argo Float {float_name_1} (RED) v {float_name_2} TS Graph")

    # Adjust spacing, show plot
    plt.tight_layout()
    plt.show()

def pres_v_var_all(df_PRESs, df_VARs, df_JULD, df_prof_nums, compare_var, float_name):

    # Make graph
    fig, ax = plt.subplots()
   
    # Normalize df_JULD for color mapping
    norm = plt.Normalize(vmin=df_JULD.min(), vmax=df_JULD.max())
    cmap = plt.get_cmap('jet')

    # Store profile nums
    selected_profiles = set()
    # Store lines for cursor connection
    lines = []
    hovered_line = None

    # Plot each profile individually
    for i in range(df_PRESs.shape[0]):
        color = cmap(norm(df_JULD[i]))  # Assign color based on date
        line, = ax.plot(df_VARs[i, :], df_PRESs[i, :], color=color, alpha=0.7)
        line.profile_number = df_prof_nums[i]  # Attach profile number to the line
        line.juld_date = datetime.datetime(1950, 1, 1) + datetime.timedelta(days=float(df_JULD[i]))  # Attach date
        lines.append(line)

    # Invert y-axis and add grid
    plt.gca().invert_yaxis()
    plt.grid(visible=True)

    # Add colorbar with date formatter
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # ScalarMappable needs this, even if not used directly

    cbar = plt.colorbar(sm, ax=ax, label='Date')
    # Set the tick locations and labels for the colorbar based on df_JULD
    cbar_ticks = np.linspace(df_JULD.min(), df_JULD.max(), num=5)
    cbar.set_ticks(cbar_ticks)
    # Convert colorbar ticks to regular dates
    cbar_labels = [datetime.datetime(1950, 1, 1) + (datetime.timedelta(days=float(juld)))  for juld in cbar.get_ticks()]
    cbar.ax.set_yticklabels([dt.strftime('%Y-%m-%d') for dt in cbar_labels])

    # Use mplcursors to display profile numbers on hover
    cursor = mplcursors.cursor(lines, hover=True)
    # Attach profile numbers to each segment
    def annotate_hover(sel):
        nonlocal hovered_line
        hovered_line = sel.artist  # Get the line that was hovered over
        profile_number = hovered_line.profile_number  # Retrieve the attached profile number
        juld_date = hovered_line.juld_date.strftime('%Y-%m-%d')  # Format the date
        sel.annotation.set_text(f"Profile: {profile_number}\nDate: {juld_date}")
    cursor.connect("add", annotate_hover)

    # Add click event
    def on_click(event):
        nonlocal selected_profiles, hovered_line
        if hovered_line:
            profile_number = hovered_line.profile_number
            if profile_number in selected_profiles:
                selected_profiles.remove(profile_number)
                print(f"Removing {profile_number}")
            else:
                selected_profiles.add(profile_number)
                print(f"Adding {profile_number}")
            print(f"Selected Profiles: {selected_profiles}")
            print("=====================" + "====" * len(selected_profiles))
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Set axis labels
    plt.ylabel("Pressure")
    if compare_var == 'PSAL':
        plt.xlabel("Salinity")
    elif compare_var == 'TEMP':
        plt.xlabel("Temperature")
    else:
        raise ValueError("Please specify 'compare_var' as 'PSAL' or 'TEMP'")

    # Set title
    plt.title(f"Argo Float {float_name} PRES-{compare_var} Graph")
    plt.show()

    return selected_profiles

def flag_point_data_graphs(var, PRES, data_type, qc_arr, profile_num, date, ax=None, figure= None):

    print_multiplot = False
    if ax is None:
        # Create the figure and axes
        fig, ax = plt.subplots()
    else:
        print_multiplot = True
        fig = figure
    
    colors = []
    selected_points = []
    for qc in qc_arr:
        if qc == 4:                 # bad
            colors.append('red')
            selected_points.append(4)
        elif qc == 3:               # prob bad
            colors.append('orange')
            selected_points.append(3)
        elif qc == 2:               # prob good
            colors.append('aqua')
            selected_points.append(2)
        else:                       # qc == 1, val is good
            colors.append('green')
            selected_points.append(1)
            
    # Plot good points (green)
    if data_type == "PRES":
        # Scatter plot for good and bad points
        scatter_plt = ax.scatter(np.arange(len(PRES)), PRES, color=colors, s=30, alpha=0.9, label='Good Data')
    else:
        ax.plot(var, PRES, color='blue', linewidth=2)
        scatter_plt = ax.scatter(var, PRES, color=colors, s=30, alpha=0.9)

    # Invert y-axis and add grid
    ax.invert_yaxis()
    ax.grid(visible=True)

    # Hover functionality
    cursor = mplcursors.cursor([scatter_plt], hover=True)
    def annotate_hover(sel):
        x, y = sel.target
        sel.annotation.set_text(f"{data_type}: {x:.2f}\nPRES: {y}")
    cursor.connect("add", annotate_hover)

    # Click event
    def on_click(event):
        nonlocal selected_points
        for scatter, graph_colors in [(scatter_plt, colors)]:
            cont, ind = scatter.contains(event)
            if cont:
                clicked_idx = ind["ind"][0] 
                # Get org color of point
                org_color = graph_colors[clicked_idx]
                # Cycle through color options
                if org_color == 'red':
                    graph_colors[clicked_idx] = 'orange'
                    selected_points[clicked_idx] = 3
                elif org_color == 'orange':
                    graph_colors[clicked_idx] = 'aqua'
                    selected_points[clicked_idx] = 2
                elif org_color == 'aqua':
                    graph_colors[clicked_idx] = 'green'
                    selected_points[clicked_idx] = 1
                else:
                    graph_colors[clicked_idx] = 'red'
                    selected_points[clicked_idx] = 4
                
                # Update the color of the clicked point
                scatter.set_color(graph_colors)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Custom legend elements
    custom_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),    # Both bad
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10),   # Salinity bad
        Line2D([0], [0], marker='o', color='w', markerfacecolor='aqua', markersize=10), # Temperature bad
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10)   # Good data
    ]
    # Add legend to the plot
    ax.legend(
        custom_legend,
        ["Bad", "Probably Bad", "Probably Good", "Good"],  # Custom labels
        loc='lower left', title="Data Quality"
    )

    if data_type == "PRES":
        # Add labels and title
        ax.set_xlabel("Index")
        ax.set_ylabel('Pressure')
        if print_multiplot == False:
            ax.set_title(f"Flag Point Pressure Graph for Profile: {profile_num} on {from_julian_day(float(date)).date()}")
    else:
        # Add labels and title
        ax.set_xlabel(data_type)
        ax.set_ylabel('Pressure')
        if print_multiplot == False:
            ax.set_title(f"Flag Point PRES v {data_type} for Profile: {profile_num} on {from_julian_day(float(date)).date()}")
    
    if print_multiplot == False:
        plt.show()

    return selected_points
    
def merge_ranges(ranges):
    # Sort the ranges by their start values
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    
    # Initialize an empty list to store merged ranges
    merged = []
    
    for start, end in sorted_ranges:
        # If merged is empty or the current range does not overlap with the last range in merged
        if not merged or merged[-1][1] < start:
            merged.append((start, end))
        else:
            # Merge the current range with the last range in merged
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    
    return merged

def flag_range_data_graphs(var, PRES, data_type, qc_arr, profile_num, date):

    # Separate points based on QC flags
    bad_mask = (qc_arr == 4) | (qc_arr == 3)

    # Array to store selected points
    selected_points = [] 

    # Create the figure and axes
    fig, ax = plt.subplots()

    # init colors in arr
    colors = ['green' for _ in range(len(PRES))]
    for i in np.where(bad_mask == True)[0]:
        colors[i] = 'red'
    # copy org color arr
    org_colors = copy.deepcopy(colors)
  
    # Plot points
    if data_type == 'PRES':
        scatter_plt = ax.scatter(np.arange(len(PRES)), PRES, color=colors, s=30, alpha=0.9)
    else:
        ax.plot(var, PRES, color='blue', linewidth=2)  # Line plot
        scatter_plt = ax.scatter(var, PRES, color=colors, s=30, alpha=0.9)

    # Invert y-axis and add grid
    plt.gca().invert_yaxis()
    plt.grid(visible=True)

    # Hover functionality
    cursor = mplcursors.cursor([scatter_plt], hover=True)
    def annotate_hover(sel):
        x, y = sel.target
        sel.annotation.set_text(f"{data_type}: {x:.2f}\nPRES: {y}")
    cursor.connect("add", annotate_hover)

    # Click event
    def on_click(event):
        # Tells function to use a nonlocal selected points 
        nonlocal selected_points, org_colors
        for scatter, subset_colors in [(scatter_plt, colors)]:
            cont, ind = scatter.contains(event)
            if cont:
                clicked_idx = ind["ind"][0]  
                complete_incomplete_pair = False
                if data_type == "PRES":
                    var_val = PRES[clicked_idx]
                else: 
                    var_val = var[clicked_idx]

                for i, (p1, p2) in enumerate(selected_points):
                    # Check if clicked point is in range of a pair
                    if p1 is not None and p2 is not None:
                        if clicked_idx in np.arange(p1, p2 + 1):
                            # If it is, remove the entire range of color and remove the pair
                            selected_points.pop(i)
                            for val_in_range in np.arange(p1, p2 + 1):
                                # set color back to orginal color 
                                subset_colors[val_in_range] = org_colors[val_in_range]
                            # Set color and update 
                            scatter.set_color(subset_colors)
                            fig.canvas.draw_idle()
                            # exit after removal
                            print(f"Deleting range: {p1} - {p2}")
                            return 
                # If here, then clicked point is not in range of pair
                # Add the clicked point to a new or incomplete pair
                if not selected_points or selected_points[-1][1] is not None:
                    # Check if org color is red
                    if subset_colors[clicked_idx] == 'red':
                        # If it is mark it as green              
                        subset_colors[clicked_idx] = 'green'
                        print(f"(index, val): ({clicked_idx}, {var_val}) marked as good")
                    else:
                        # Start a new pair
                        selected_points.append((clicked_idx, None))
                        # Change color of that point 
                        subset_colors[clicked_idx] = 'red'
                        print(f"Starting new pair")
                else:
                    # Complete the last incomplete pair
                    # Check to see if it is the same
                    if clicked_idx == selected_points[-1][0]:
                        # remove that val
                        selected_points.pop(-1)
                        # change color to org color
                        org_color = subset_colors[clicked_idx]
                        if org_color == 'red':
                            subset_colors[clicked_idx] = 'green'
                        else:
                            subset_colors[clicked_idx] = 'red'
                        # Update the color of the clicked point
                        scatter.set_color(subset_colors)
                        fig.canvas.draw_idle()
                        print(f"Point deleted")
                        return
                    # Check to see if clicked val is before or after the last incomplete pair val
                    # If it happened after
                    elif clicked_idx > selected_points[-1][0]:
                        complete_incomplete_pair = True
                        # add the point and color as normal
                        selected_points[-1] = (selected_points[-1][0], clicked_idx)
                        # Change color to red 
                        for val_in_range in np.arange(selected_points[-1][0], selected_points[-1][1] + 1):
                            subset_colors[val_in_range] = 'red'
                        print(f"Completing pair: {selected_points[-1]}")
                    else:
                        complete_incomplete_pair = True
                        # switch the pair around and color as normal
                        selected_points[-1] = (clicked_idx, selected_points[-1][0])
                        # Change color to red 
                        for val_in_range in np.arange(selected_points[-1][0], selected_points[-1][1] + 1):
                            subset_colors[val_in_range] = 'red'
                        print(f"Completing pair: {selected_points[-1]}")

                # check to see if we can merge any ranges 
                if complete_incomplete_pair == True:
                    selected_points_copy = copy.deepcopy(selected_points) # avoid modifying in place
                    selected_points = merge_ranges(selected_points_copy)

                # Update the color of the clicked point
                scatter.set_color(subset_colors)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)

    # Add labels and title
    if data_type == "PRES":
        plt.xlabel("Index")
        plt.ylabel('Pressure')
        plt.title(f"Flag Range Pressure Graph for Profile: {profile_num} on {from_julian_day(float(date)).date()}")
    else:
        plt.xlabel(data_type)
        plt.ylabel('Pressure')
        plt.title(f"Flag Range PRES v {data_type} for Profile: {profile_num} on {from_julian_day(float(date)).date()}")

    plt.show()

    return selected_points

def flag_TS_data_graphs(sal, temp, date, lons, lats, pres, profile_num, temp_adjusted_qc, psal_adjusted_qc, ax = None):
    
    print_multiplot = False

    sal_copy = copy.deepcopy(sal)
    temp_copy = copy.deepcopy(temp)
    
    # Convert data
    if (not np.isnan(lats)) and (not np.isnan(lons)):
        sal_copy = gsw.conversions.SA_from_SP(sal, temp, lons, lats)
        temp_copy = gsw.conversions.CT_from_t(sal, temp, pres)
    else:
        sal_copy = sal
        temp_copy = temp

    # Define salinity and temperature bounds for the contour plot
    smin = np.nanmin(sal_copy) - 1
    smax = np.nanmax(sal_copy) + 1
    tmin = np.nanmin(temp_copy) - 1
    tmax = np.nanmax(temp_copy) + 1

    # Calculate grid cells needed in x and y dimensions
    xdim = int(np.ceil((smax - smin) / 0.1))
    ydim = int(np.ceil((tmax - tmin)))
    dens = np.zeros((ydim, xdim))

    # Create temp and salt vectors for density
    ti = np.linspace(0, ydim, ydim) + tmin
    si = np.linspace(1, xdim, xdim) * 0.1 + smin

    # Fill grid with densities
    for j in range(ydim):
        for i in range(xdim):
            dens[j, i] = gsw.rho(si[i], ti[j], 0)

    # Subtract 1000 to convert to sigma-t
    dens = dens - 1000

    # init colors arr
    colors = []
    # init selected points arr
    selected_points = []
    for sal_qc, temp_qc in zip(psal_adjusted_qc, temp_adjusted_qc):
        if (sal_qc == 4 or sal_qc == 3) and (temp_qc == 4 or temp_qc == 3):
            colors.append('red')      # Both salinity and temperature are bad
            selected_points.append(4)
        elif sal_qc == 4 or sal_qc == 3:
            colors.append('aqua')   # Only salinity is bad
            selected_points.append(3)
        elif temp_qc == 4 or temp_qc == 3:
            colors.append('violet')   # Only temperature is bad
            selected_points.append(2)
        else:
            colors.append('green')    # Neither is bad
            selected_points.append(1) 

    # Plot data
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        print_multiplot = True

    CS = ax.contour(si, ti, dens, linestyles='dashed', colors='k')
    ax.clabel(CS, fontsize=12, inline=1, fmt='%.2f')
    # Plot TS line
    ts_line = ax.plot(sal_copy, temp_copy, color='dimgrey', linewidth=2, zorder=1)
    # Plot the salinity-temperature relationship as a scatter plot
    scatter_plt = ax.scatter(sal_copy, temp_copy, color=colors, s=20, alpha=0.9, zorder = 2)
  
    # Use mplcursors to display profile numbers on hover
    cursor = mplcursors.cursor(scatter_plt, hover=True)
    def annotate_hover(sel):
        s, t = sel.target
        sel.annotation.set_text(f"Temperature: {t:.2f} C\nSalinity: {s:.2f} PSU")
    cursor.connect("add", annotate_hover)

    # click event
    def on_click(event):
        # Tells function to use a nonlocal selected points 
        nonlocal selected_points
        for scatter, graph_colors in [(scatter_plt, colors)]:
            cont, ind = scatter.contains(event)
            if cont:
                clicked_idx = ind["ind"][0]  
                # Get org color of point
                org_color = graph_colors[clicked_idx]
                # Cycle through color options
                if org_color == 'red':
                    graph_colors[clicked_idx] = 'aqua'
                    selected_points[clicked_idx] = 3
                elif org_color == 'aqua':
                    graph_colors[clicked_idx] = 'violet'
                    selected_points[clicked_idx] = 2
                elif org_color == 'violet':
                    graph_colors[clicked_idx] = 'green'
                    selected_points[clicked_idx] = 1
                else: # color is green
                    graph_colors[clicked_idx] = 'red'
                    selected_points[clicked_idx] = 4
                
                # Update the color of the clicked point
                scatter.set_color(graph_colors)
                fig.canvas.draw_idle()
    if print_multiplot == False:
        fig.canvas.mpl_connect('button_press_event', on_click)

    # Set x and y limits, labels, and title
    ax.set_xlim([smin + 0.75, smax - 0.75])
    ax.set_ylim([tmin + 0.75, tmax - 0.75])

    # Custom legend elements
    custom_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),    # Both bad
        Line2D([0], [0], marker='o', color='w', markerfacecolor='aqua', markersize=10),   # Salinity bad
        Line2D([0], [0], marker='o', color='w', markerfacecolor='violet', markersize=10), # Temperature bad
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10)   # Good data
    ]
    # Add legend to the plot
    ax.legend(
        custom_legend,
        ["Both bad", "Salinity bad", "Temperature bad", "Good data"],  # Custom labels
        loc='lower left', title="Data Quality"
    )

    ax.set_xlabel('Salinity (PSU)')
    ax.set_ylabel('Temperature (degC)')
 
    if print_multiplot == False:
        ax.set_title(f"TS Graph for Profile: {profile_num} on {from_julian_day(float(date)).date()}")
        plt.tight_layout()
        plt.show()

    return selected_points

def deep_section_var_all(pressure, dates, COMP_vars, float_num, deep_section_compare_var):

    # Create a meshgrid for dates and pressure, matching the temperature data
    X, Y = np.meshgrid(dates, np.linspace(np.nanmin(pressure), np.nanmax(pressure), pressure.shape[1]), indexing='ij')

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create plots
    contour = ax.pcolormesh(X, Y, COMP_vars, cmap='jet')
    ax.contour(X, Y, COMP_vars, colors = 'black')

    # Invert the y-axis to have depth increasing downwards
    ax.invert_yaxis()

    # Convert julian times to datetime objs and set tick labels
    date_formatter = DateFormatter('%Y-%m-%d')
    ticks =[datetime.datetime(1950, 1, 1) + (datetime.timedelta(days=float(juld)))  for juld in ax.get_xticks()]
    ax.xaxis.set_major_formatter(date_formatter)
    ax.xaxis.set_ticklabels([dt.strftime('%Y-%m-%d') for dt in ticks])

    # Add a colorbar
    cbar = plt.colorbar(contour, ax=ax)
    if deep_section_compare_var == "TEMP":
        cbar.set_label('Temperature In-Situ (°C)')
    elif deep_section_compare_var == "PSAL":
        cbar.set_label('Practical Salinity (psu)')
    else:
        raise Exception("deep_section_compare_var must be TEMP or PSAL not {}".format(deep_section_compare_var))

    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Index')
    ax.set_title(f'Argo Float {float_num} Deep Section {deep_section_compare_var} Graph')

    plt.show()

def TS_graph_single_dataset_all_profile(df_SALs, df_TEMPs, df_JULD, df_LONs, df_LATs, df_PRESs, df_prof_nums, float_name):
    
    selected_profiles = set()
    segment_index = None

    df_SALs_copy = copy.deepcopy(df_SALs)
    df_TEMPs_copy = copy.deepcopy(df_TEMPs)
    
    # Convert data for ref dataset
    for i in np.arange(df_SALs.shape[0]):    # number of profiles
        df_SALs_copy[i, :] = gsw.conversions.SA_from_SP(df_SALs[i, :], df_TEMPs[i, :], df_LONs[i], df_LATs[i])
        df_TEMPs_copy[i, :] = gsw.conversions.CT_from_t(df_SALs[i, :], df_TEMPs[i, :], df_PRESs[i, :])
    
    # Define salinity and temperature bounds for the contour plot
    smin = np.nanmin(df_SALs_copy) - 1
    smax = np.nanmax(df_SALs_copy) + 1
    tmin = np.nanmin(df_TEMPs_copy) - 1
    tmax = np.nanmax(df_TEMPs_copy) + 1

    # Calculate grid cells needed in x and y dimensions
    xdim = int(np.ceil((smax - smin) / 0.1))
    ydim = int(np.ceil((tmax - tmin)))
    dens = np.zeros((ydim, xdim))

    # Create temp and salt vectors for density
    ti = np.linspace(0, ydim, ydim) + tmin
    si = np.linspace(1, xdim, xdim) * 0.1 + smin

    # Fill grid with densities
    for j in range(ydim):
        for i in range(xdim):
            dens[j, i] = gsw.rho(si[i], ti[j], 0)

    # Subtract 1000 to convert to sigma-t
    dens = dens - 1000

    ref_segments = [np.column_stack((df_SALs[i, :], df_TEMPs[i, :])) for i in range(df_SALs.shape[0])]
    ref_lc = LineCollection(ref_segments, array=df_JULD, cmap='jet', alpha=0.7)
    ref_lc.set_linewidth(1)

    # Attach the profile numbers to the LineCollection
    ref_lc.profile_numbers = df_prof_nums

    # Plot data
    fig, ax = plt.subplots(figsize=(10, 6))
    CS = plt.contour(si, ti, dens, linestyles='dashed', colors='k')
    plt.clabel(CS, fontsize=12, inline=1, fmt='%.2f')

    # Add LineCollection to plot
    ax = plt.gca()
    ax.add_collection(ref_lc)

    # Add colorbar with date
    cbar = plt.colorbar(ref_lc, ax=ax, label='Date')
    date_formatter = DateFormatter('%Y-%m-%d')

    # Convert colorbar ticks to regular dates
    cbar_ticks = [datetime.datetime(1950, 1, 1) + (datetime.timedelta(days=float(juld)))  for juld in cbar.get_ticks()]
    cbar.ax.yaxis.set_major_formatter(date_formatter)
    cbar.set_ticklabels([dt.strftime('%Y-%m-%d') for dt in cbar_ticks])

    # Set x and y limits, labels, and title
    plt.xlim([smin + 0.75, smax - 0.75])
    plt.ylim([tmin + 0.75, tmax - 0.75])
    plt.xlabel('Salinity (PSU)')
    plt.ylabel('Temperature (degC)')
    plt.title(f"Argo Float {float_name} TS Graph")
    plt.tight_layout()

    # Use mplcursors to display profile numbers on hover
    cursor = mplcursors.cursor(ref_lc, hover=True)
    
    # Attach profile numbers to each segment
    def annotate_hover(sel):
        nonlocal segment_index
        segment_index = sel.index[0]  # Index of the segment in LineCollection
        profile_number = ref_lc.profile_numbers[segment_index]  # Get the corresponding profile number
        sel.annotation.set_text(f"Profile: {profile_number}")
    
    def on_click(event):
        nonlocal selected_profiles, segment_index
        if segment_index:
            profile_number = ref_lc.profile_numbers[segment_index]
            if profile_number in selected_profiles:
                selected_profiles.remove(profile_number)
                print(f"Removing {profile_number}")
            else:
                selected_profiles.add(profile_number)
                print(f"Adding {profile_number}")
            print(f"Selected Profiles: {selected_profiles}")
            print("=====================" + "====" * len(selected_profiles))
    fig.canvas.mpl_connect('button_press_event', on_click)

    cursor.connect("add", annotate_hover)
    plt.show()

    return selected_profiles

def del_bad_points(PRES_ADJUSTED_QC, TEMP_ADJUSTED_QC, PSAL_ADJUSTED_QC,
                   PSAL_ADJUSTED, TEMP_ADJUSTED, PRES_ADJUSTED):
    
    # Identify bad values (marked as 4) in each QC array
    pres_bad = (PRES_ADJUSTED_QC == 4)
    temp_bad = (TEMP_ADJUSTED_QC == 4)
    psal_bad = (PSAL_ADJUSTED_QC == 4)
    
    # Combine masks: mark as bad if any of the arrays have bad values
    combined_bad_mask = pres_bad | temp_bad | psal_bad
    
    # Replace bad values with NaN, preserving the original shape
    PSAL_ADJUSTED = np.where(combined_bad_mask, np.nan, PSAL_ADJUSTED)
    TEMP_ADJUSTED = np.where(combined_bad_mask, np.nan, TEMP_ADJUSTED)
    PRES_ADJUSTED = np.where(combined_bad_mask, np.nan, PRES_ADJUSTED)
    
    return PSAL_ADJUSTED, TEMP_ADJUSTED, PRES_ADJUSTED

def main():
   
    shallow_cutoff = 1 # code in this feature
    # generate graphs for F9186 or F10051
    float_num = "F10051"

    nc_filepath = "c:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\argo_to_nc\\F10051_0"
    float_name_1 = "F10051_0"
    argo_data_1 = read_intermediate_nc_file(nc_filepath)
    
    nc_filepath = "c:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\argo_to_nc\\F10051_1"
    float_name_2 = "F10051_1"
    argo_data_2 = read_intermediate_nc_file(nc_filepath)
    
    # Make sure arr vals marked as bad are reflected across ALL arrays
    """
    PSAL_ADJUSTED, TEMP_ADJUSTED, PRES_ADJUSTED = del_bad_points(PRES_ADJUSTED_QC, TEMP_ADJUSTED_QC, PSAL_ADJUSTED_QC,
                                                                 PSAL_ADJUSTED, TEMP_ADJUSTED, PRES_ADJUSTED)
    PSAL_ADJUSTED_2, TEMP_ADJUSTED_2, PRES_ADJUSTED_2 = del_bad_points(PRES_ADJUSTED_QC_2, TEMP_ADJUSTED_QC_2, PSAL_ADJUSTED_QC_2,
                                                                       PSAL_ADJUSTED_2, TEMP_ADJUSTED_2, PRES_ADJUSTED_2)
    """

    # Get rid of all nan slices
    #argo_data_1 = del_all_nan_slices(argo_data_1)
    #argo_data_2 = del_all_nan_slices(argo_data_2)

    raise Exception
    #TS_graph_double(PSAL_ADJUSTED, TEMP_ADJUSTED, JULDs, LONs, LATs, PRES_ADJUSTED, LATs_2, LONs_2, JULDs_2, PRES_ADJUSTED_2, PSAL_ADJUSTED_2, TEMP_ADJUSTED_2, float_name_1, float_name_2)
    TS_graph_single_dataset_all_profile(PSAL_ADJUSTED_2, TEMP_ADJUSTED_2, JULDs_2, LONs_2, LATs_2, PRES_ADJUSTED_2, PROFILE_NUMS_2, float_name_2)
    TS_graph_single_dataset_all_profile(PSAL_ADJUSTED, TEMP_ADJUSTED, JULDs, LONs, LATs, PRES_ADJUSTED, PROFILE_NUMS, float_name_1)
   
if __name__ == '__main__':
 
    main()