import itertools
from pathlib import Path
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
import RBRargo3_celltm
from tools import from_julian_day, read_intermediate_nc_file, to_julian_day
from matplotlib.lines import Line2D
import csv
from scipy.interpolate import interp1d

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

def flag_point_data_graphs(var, PRES, data_type, qc_arr, profile_num, date, argodata=None, ax=None, figure= None):

    print_multiplot = False
    if ax is None:
        # Create the figure and axes
        fig, ax = plt.subplots()
    else:
        print_multiplot = True
        fig = figure

    if argodata is None and data_type == "PSAL":
        raise Exception("Need argo_data param if data type is PSAL to run density inversion tests")
    
    colors = []
    selected_points = []
    edge_colors = []
    exclude = []
    for qc in qc_arr:
        if qc == 4:                 # bad
            colors.append('red')
            selected_points.append(4)
            edge_colors.append('red')
        elif qc == 3:               # prob bad
            colors.append('orange')
            selected_points.append(3)
            edge_colors.append('orange')
        elif qc == 2:               # prob good
            colors.append('aqua')
            selected_points.append(2)
            edge_colors.append('aqua')
        else:                       # qc == 1, val is good
            colors.append('green')
            selected_points.append(1)
            edge_colors.append('green')
    
    # Inversion test
    if argodata is not None:
        inversion = density_inversion_test(argodata, profile_num)
        for i in inversion:
            edge_colors[i] = 'fuchsia'

    # Plot good points (green)
    if data_type == "PRES":
        # Scatter plot for good and bad points
        scatter_plt = ax.scatter(np.arange(len(PRES)), PRES, color=colors, s=35, alpha=0.9, label='Good Data')
    else:
        ax.plot(var, PRES, color='blue', linewidth=2)
        scatter_plt = ax.scatter(var, PRES, color=colors, edgecolors=edge_colors, s=35, alpha=0.9)

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
        nonlocal selected_points, argodata, profile_num, exclude
        for scatter, graph_colors, edgecolor in [(scatter_plt, colors, edge_colors)]:
            cont, ind = scatter.contains(event)
            if cont:
                clicked_idx = ind["ind"][0] 
                # Get org color of point
                org_color = graph_colors[clicked_idx]
                # Check to see if clicked point was flagged as density inversion
                run_density_inversion = False
                if argodata is not None:
                    if edgecolor[clicked_idx] == 'fuchsia':
                        run_density_inversion = True
                # Cycle through color options
                if org_color == 'red':
                    graph_colors[clicked_idx] = 'orange'
                    selected_points[clicked_idx] = 3
                    edge_colors[clicked_idx] = 'orange'
                elif org_color == 'orange':
                    graph_colors[clicked_idx] = 'aqua'
                    selected_points[clicked_idx] = 2
                    edge_colors[clicked_idx] = 'aqua'
                elif org_color == 'aqua':
                    graph_colors[clicked_idx] = 'green'
                    selected_points[clicked_idx] = 1
                    edge_colors[clicked_idx] = 'green'
                else:
                    graph_colors[clicked_idx] = 'red'
                    selected_points[clicked_idx] = 4
                    edge_colors[clicked_idx] = 'red'

                if run_density_inversion == True:
                    # see what color we've landed on
                    # if we've marked a point as bad, we need to check it again
                    if graph_colors[clicked_idx] == 'orange' or graph_colors[clicked_idx] == 'red':
                        # add it to exclude list
                        exclude.append(clicked_idx)
                        inversion = density_inversion_test(argodata, profile_num, exclude_pts=exclude)
                        for i in inversion:
                            edgecolor[i] = 'fuchsia'

                # Update the color of the clicked point
                scatter.set_color(graph_colors)
                scatter.set_edgecolors(edgecolor)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)
    
    if print_multiplot == False:
        # Custom legend elements
        if argodata is not None:
            custom_legend = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='aqua', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor='fuchsia', markersize=9) 
            ]
            # Add legend to the plot
            ax.legend(
                custom_legend,
                ["Bad", "Probably Bad", "Probably Good", "Good", "Density Inversion"],  # Custom labels
                loc='lower left', title="Data Quality"
            )
        else: 
            custom_legend = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='aqua', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10),
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

def density_inversion_test(argo_data, prof_num, exclude_pts = None):

    i =  np.where(argo_data["PROFILE_NUMS"] == prof_num)[0][0]
    psal = copy.deepcopy(argo_data["PSAL_ADJUSTED"][i])
    temp = copy.deepcopy(argo_data["TEMP_ADJUSTED"][i])
    pres = copy.deepcopy(argo_data["PRES_ADJUSTED"][i])
    # apply QC to data
    psal[np.where(argo_data["PSAL_ADJUSTED_QC"][i] == 4)] = np.nan
    temp[np.where(argo_data["TEMP_ADJUSTED_QC"][i] == 4)] = np.nan
    pres[np.where(argo_data["PRES_ADJUSTED_QC"][i] == 4)] = np.nan
    lat = argo_data["LATs"][i]
    lon = argo_data["LONs"][i]

    # Get rid of excluded points
    if exclude_pts is not None:
        for i in exclude_pts:
            psal[i] = np.nan
            temp[i] = np.nan
            pres[i] = np.nan

    # Convert PSAL to Absolute Salinity
    abs_sal = gsw.SA_from_SP(psal, pres, lon, lat)
    # Convert TEMP to Conservative Temperature
    cons_temp = gsw.CT_from_t(abs_sal, temp, pres)
    # Calculate in-situ density
    dens = gsw.rho(abs_sal, cons_temp, pres)
 
    failed_idxs = []
    valid_idxs = np.where(~np.isnan(dens))[0]   # gets only valid dens
    for j in range(len(valid_idxs) - 1):
            idx1 = valid_idxs[j]
            idx2 = valid_idxs[j + 1]
            if dens[idx2] < dens[idx1]:
                # Mark both points involved in inversion
                failed_idxs.extend([idx1, idx2])

    failed_idxs = sorted(set(failed_idxs))  # Remove duplicates and sort
    print(f"Failed density inversion at indices: {failed_idxs}")

    return failed_idxs

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

    # Create the figure and axes
    fig, ax = plt.subplots()

    # init colors in arr
    colors = []
    selected_points = [] 
    for qc in qc_arr:
        if qc == 4:                 # bad
            colors.append('red')
        elif qc == 3:               # prob bad
            colors.append('orange')
        elif qc == 2:               # prob good
            colors.append('aqua')
        else:                       # qc == 1, val is good
            colors.append('green')
    # copy org color arr
    org_colors = copy.deepcopy(colors)
  
    # Plot points
    if data_type == 'PRES':
        scatter_plt = ax.scatter(np.arange(len(PRES)), PRES, color=colors, s=35, alpha=0.9)
    else:
        ax.plot(var, PRES, color='blue', linewidth=2)  # Line plot
        scatter_plt = ax.scatter(var, PRES, color=colors, s=35, alpha=0.9)

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

                for i, (p1, p2) in enumerate(selected_points):
                    # Check if clicked point is in range of a pair
                    if p1 is not None and p2 is not None:
                        if clicked_idx in np.arange(p1, p2 + 1):
                            # If it is, remove the entire range of color and remove the pair
                            selected_points.pop(i)
                            for val_in_range in np.arange(p1, p2 + 1):
                                # set color back to original color 
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
                    # Change color of clicked element 
                    if org_colors[clicked_idx] == 'red':
                        subset_colors[clicked_idx] = 'orange'
                    elif org_colors[clicked_idx] == 'orange':
                        subset_colors[clicked_idx] = 'aqua'
                    elif org_colors[clicked_idx] == 'aqua':
                        subset_colors[clicked_idx] = 'green'
                    else:
                        subset_colors[clicked_idx] = 'red'
                    selected_points.append((clicked_idx, None))
                    print(f"Starting new pair")
                else:
                    # Complete the last incomplete pair
                    # Check to see if it is the same
                    if clicked_idx == selected_points[-1][0]:
                        # If it is the same, cycle color options
                        if subset_colors[clicked_idx] == 'red':
                            subset_colors[clicked_idx] = 'orange'
                        elif subset_colors[clicked_idx] == 'orange':
                            subset_colors[clicked_idx] = 'aqua'
                        elif subset_colors[clicked_idx] == 'aqua':
                            subset_colors[clicked_idx] = 'green'
                        elif subset_colors[clicked_idx] == 'green':
                            subset_colors[clicked_idx] = 'red'
                        # Delete point when color has cycled to orginal color
                        if org_colors[clicked_idx] == subset_colors[clicked_idx]:
                            # remove that val
                            selected_points.pop(-1)
                            print(f"Point deleted")
                        # Update the color of the clicked point
                        scatter.set_color(subset_colors)
                        fig.canvas.draw_idle()
                        return
                    # Check to see if clicked val is before or after the last incomplete pair val
                    # If it happened after
                    elif clicked_idx > selected_points[-1][0]:
                        complete_incomplete_pair = True
                        # add the point and color as normal
                        selected_points[-1] = (selected_points[-1][0], clicked_idx)
                        # Change color
                        for val_in_range in np.arange(selected_points[-1][0], selected_points[-1][1] + 1):
                            subset_colors[val_in_range] = subset_colors[selected_points[-1][0]]
                        print(f"Completing pair: {selected_points[-1]}")
                    else:
                        complete_incomplete_pair = True
                        # switch the pair around and color as normal
                        selected_points[-1] = (clicked_idx, selected_points[-1][0])
                        # Change color
                        for val_in_range in np.arange(selected_points[-1][0], selected_points[-1][1] + 1):
                            subset_colors[val_in_range] = subset_colors[selected_points[-1][1]]
                        print(f"Completing pair: {selected_points[-1]}")

                # # check to see if we can merge any ranges 
                if complete_incomplete_pair == True:
                    selected_points_copy = copy.deepcopy(selected_points) # avoid modifying in place
                    selected_points = merge_ranges(selected_points_copy)

                # Update the color of the clicked point
                scatter.set_color(subset_colors)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)

    # Custom legend elements
    custom_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10), 
        Line2D([0], [0], marker='o', color='w', markerfacecolor='aqua', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10)   
    ]
    # Add legend to the plot
    ax.legend(
        custom_legend,
        ["Bad", "Probably Bad", "Probably Good", "Good"],  # Custom labels
        loc='lower left', title="Data Quality"
    )

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

    return colors

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
    plt.grid(True)
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


def make_thermal_inertia_graph():

    fp = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\argo_to_nc\\F10051_after_first_time_run"
    argo_data = read_intermediate_nc_file(fp)

    fig, ax = plt.subplots()

    norm = plt.Normalize(vmin=argo_data["JULDs"].min(), vmax=argo_data["JULDs"].max())
    cmap = plt.get_cmap('jet')

    for i in np.arange(argo_data["TEMPs"].shape[0]):

        color = cmap(norm(argo_data["JULDs"][i]))  # Assign color based on date
        plt.plot(argo_data["TEMPs"][i] - argo_data["TEMP_CNDCs"][i],  argo_data["PRES_ADJUSTED"][i], linewidth=0.5, color = color)
        #plt.plot(argo_data["TEMPs"][i],  argo_data["PRES_ADJUSTED"][i], linewidth=0.5, color = color)
        #plt.plot(argo_data["TEMP_CNDCs"][i],  argo_data["PRES_ADJUSTED"][i], linewidth=0.5, color = color)

    # Add colorbar with date formatter
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # ScalarMappable needs this, even if not used directly
    cbar = plt.colorbar(sm, ax=ax, label='Date')
    # Set the tick locations and labels for the colorbar based on df_JULD
    cbar_ticks = np.linspace(argo_data["JULDs"].min(), argo_data["JULDs"].max(), num=5)
    cbar.set_ticks(cbar_ticks)
    # Convert colorbar ticks to regular dates
    cbar_labels = [datetime.datetime(1950, 1, 1) + (datetime.timedelta(days=float(juld)))  for juld in cbar.get_ticks()]
    cbar.ax.set_yticklabels([dt.strftime('%Y-%m-%d') for dt in cbar_labels])

    plt.xlabel('TEMP - TEMP_CNDC °C')
    plt.ylabel('Pressure Adjusted (dbar)')
    plt.title('F10051 TEMP - TEMP_CNDC Difference vs Pressure')

    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def read_csv_file_for_thermal_inertia_graph_with_timestamps(input_filepath):
    
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
    
    PRESs, TEMPs, SALs, CNDCs, TEMP_CNDCs, COUNTs, TIMESTAMPs, PROF_NUMs = [], [], [], [], [], [], [], []
    
    # Process each profile only if both files are present
    for profile_num, file_paths in files_dictionary.items():

        if "science_log" in file_paths and "system_log" in file_paths:
            # Initialize variables
            pressures, temps, sals, cndc, temp_cndc, counts, timestamps = [], [], [], [], [], [], []
            with open(file_paths["science_log"], mode='r') as sci_file:
                reader = csv.reader(sci_file)
                PTSCIinfo = []
                for row in reader:
                    if(row[0] == "LGR_CP_PTSCI"):
                        PTSCIinfo.append(row)  
            # Process data from science file
            for row in PTSCIinfo:
                pressures.append(float(row[2]))
                temps.append(float(row[3]))
                sals.append(float(row[4]))
                cndc.append(float(row[5]))
                temp_cndc.append(float(row[6]))
                counts.append(int(row[-1]))
                timestamps.append(row[1])

            if len(PTSCIinfo) > 2:
                # convert timestamps
                times = np.array([datetime.datetime.strptime(ts, "%Y%m%dT%H%M%S") for ts in timestamps])
                # Compute seconds relative to the first timestamp
                time_deltas = np.array([(dt - times[0]).total_seconds() for dt in times])
        
                # convert arrs to np arrs
                pressures = np.asarray(pressures)
                temps = np.asarray(temps)
                sals = np.asarray(sals)
                cndc = np.asarray(cndc)
                temp_cndc = np.asarray(temp_cndc)
                counts = np.asarray(counts)
                timestamps = np.asarray(time_deltas)
    
                # Init vars for sys file
                offset = None
                with open(file_paths["system_log"], mode='r') as sys_file:
                    for line in sys_file:
                        if 'surface pressure offset' in line:
                            line = line.split(' ')
                            offset = line[-2]
                if offset is None:
                    print(f"Profile {profile_num} is missing 'surface pressure offset' in system log")
                else:
                    pressures =  pressures - float(offset)
                    # append data to overall arr
                    PRESs.append(pressures)
                    TEMPs.append(temps)
                    SALs.append(sals)
                    CNDCs.append(cndc)
                    TEMP_CNDCs.append(temp_cndc)
                    COUNTs.append(counts)
                    TIMESTAMPs.append(timestamps)
                    PROF_NUMs.append(profile_num)
            else:
                print(f"Skipping profile {profile_num}: PTSCI info missing")
        else:
           print(f"Skipping profile {profile_num}: Missing required files.")
    
    """
    PRESs = np.squeeze(np.array(list(itertools.zip_longest(*PRESs, fillvalue=np.nan))).T)
    TEMPs = np.squeeze(np.array(list(itertools.zip_longest(*TEMPs, fillvalue=np.nan))).T)
    SALs = np.squeeze(np.array(list(itertools.zip_longest(*SALs, fillvalue=np.nan))).T)
    CNDCs = np.squeeze(np.array(list(itertools.zip_longest(*CNDCs, fillvalue=np.nan))).T)
    TEMP_CNDCs = np.squeeze(np.array(list(itertools.zip_longest(*TEMP_CNDCs, fillvalue=np.nan))).T)
    COUNTs = np.squeeze(np.array(list(itertools.zip_longest(*COUNTs, fillvalue=np.nan))).T)
    TIMESTAMPs = np.squeeze(np.array(list(itertools.zip_longest(*TIMESTAMPs, fillvalue=np.nan))).T)
    """

    return PRESs, TEMPs, SALs, CNDCs, TEMP_CNDCs, COUNTs, TIMESTAMPs, PROF_NUMs

def make_thermal_inertia_graph_with_timestamps():

    filepath = "C:\\Users\\szswe\\Downloads\\graphs_thermal_inertia\\DATA\\F10051_all_data_for_temp_graphs_csv"
    # filepath = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\RAW_DATA\\F9186_raw_csv"
    PRESs, TEMPs, SALs, CNDCs, TEMP_CNDCs, COUNTs, TIMESTAMPs = read_csv_file_for_thermal_inertia_graph_with_timestamps(filepath)

    # Read in intermediate NETCDF files for QC arrs
    fp = "C:\\Users\\szswe\\Downloads\\graphs_thermal_inertia\\DATA\\F10051_after_visual_inspection_for_temp_graphs"
    argo_data = read_intermediate_nc_file(fp)
    # Get rid of bad points in QC arr
    mask = (argo_data["TEMP_ADJUSTED_QC"] == 3) | (argo_data["TEMP_ADJUSTED_QC"] == 4)
    for i in np.arange(0, len(mask) - 1):
        nan_index = np.where(~np.isnan(argo_data["PRESs"][i, :]))[0][-1] + 1
        TEMPs[i][mask[i, :nan_index] == True] = np.nan
        TEMP_CNDCs[i][mask[i, :nan_index] == True] = np.nan
    # Get rid of points where count is too high
    for i in np.arange(0, len(COUNTs)):
        TEMPs[i][np.where(COUNTs[i] > 50)[0]] = np.nan
        TEMP_CNDCs[i][np.where(COUNTs[i] > 50)[0]] = np.nan

    fig, ax = plt.subplots()
   
    # Define the uniform time grid based on the maximum time across all profiles
    all_differences = []
    max_time = max(time_seq[-1] for time_seq in TIMESTAMPs)
    uniform_time_mean = np.arange(0, max_time, 10)
    
    for i in np.arange(0, len(TIMESTAMPs)):

        time_seconds = TIMESTAMPs[i]

        #   Create a uniform time grid for this row
        #uniform_time = np.arange(0, time_seconds[-1], 10)
        #   Interpolate TEMP_CNDCs onto the uniform time grid
        #temp_cndc_interp = np.interp(uniform_time, time_seconds - 245, TEMP_CNDCs[i])
        #temp_interp = np.interp(uniform_time, time_seconds, TEMPs[i])
        #plt.plot(temp_interp - temp_cndc_interp, uniform_time, linewidth=0.5) 

        temp_cndc_interp = np.interp(uniform_time_mean, time_seconds - 150, TEMP_CNDCs[i])
        temp_interp = np.interp(uniform_time_mean, time_seconds, TEMPs[i])
        difference = temp_interp - temp_cndc_interp
        all_differences.append(difference)

    # find the mean difference 
    all_differences = np.array(all_differences)
    mean_difference = np.nanmean(all_differences, axis=0)
    plt.plot(mean_difference, uniform_time_mean, linewidth=1.5)

    # Formatting
    plt.ylabel("Time (seconds)")
    #plt.xlabel("TEMP - TEMP_CNDC °C'")
    #plt.title("TEMP - TEMP_CNDC interpolated onto uniform time grid")
    plt.xlabel('Mean TEMP - TEMP_CNDC')
    plt.title('Mean Temperature Difference Over Time')
    plt.grid()
    plt.show()

def single_prof_datasnapshot(profile_num, argo_data, PSAL_ADJUSTED_Padj_CTM):
    
    i = np.where(argo_data["PROFILE_NUMS"] == profile_num)[0][0]

    sal_arr = np.squeeze(PSAL_ADJUSTED_Padj_CTM[i])
    temp_arr = argo_data["TEMP_ADJUSTED"][i]
    pres_arr = argo_data["PRES_ADJUSTED"][i]
    temp_qc = argo_data["TEMP_ADJUSTED_QC"][i]
    psal_qc = argo_data["PSAL_ADJUSTED_QC"][i]
    lon = argo_data["LONs"][i]
    lat = argo_data["LATs"][i]
    juld = argo_data["JULDs"][i]

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Populate the top-left subplot
    selected_points_temp = flag_point_data_graphs(temp_arr, pres_arr, "TEMP", temp_qc, argo_data["PROFILE_NUMS"][i], juld, ax=axs[0, 0], figure=fig)

    # Populate the top-right subplot
    selected_points_psal = flag_point_data_graphs(sal_arr, pres_arr, "PSAL", psal_qc, argo_data["PROFILE_NUMS"][i], juld, ax=axs[0, 1], figure=fig)

    # Populate the bottom-left subplot
    flag_TS_data_graphs(sal_arr, temp_arr, juld, lon, lat, pres_arr, profile_num, temp_qc, psal_qc, ax=axs[1, 0])

    # Fill bottom-right subplot with text
    timestamp = from_julian_day(float(juld))
    axs[1, 1].text(0.5, 0.7, f'Data Snapshot of Profile: {profile_num}', fontsize=12, ha='center', va='center')
    axs[1, 1].text(0.5, 0.5, f'Datetime of Profile: {timestamp.date()} {timestamp.strftime('%H:%M:%S')}', fontsize=12, ha='center', va='center')
    axs[1, 1].text(0.5, 0.3, f'Lat: {lat:.2f} Lon: {lon:.2f}', fontsize=12, ha='center', va='center')
    axs[1, 1].text(0.5, 0.1, f'Flag QC-point feature enabled for TEMP + PSAL graphs', fontsize=12, ha='center', va='center')
    axs[1, 1].axis('off')

    axs[1,0].grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def compute_cellTM():
    
    # Read in intermediate NETCDF files for QC arrs
    # fp = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\argo_to_nc\\F10051_after_visual_inspection"
    fp = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\csv_to_nc\\F9186_after_vi_new"
    float_num = "F9186_ADJUSTED_SAL"
    use_timestamps = True
    argo_data = read_intermediate_nc_file(fp)
 
    cell_tms = []
    for i in np.arange(len(argo_data["NB_SAMPLE_CTD"])):
        nan_index = np.where(~np.isnan(argo_data["PRESs"][i, :]))[0][-1] + 1
        
        if use_timestamps == True:
            # NO FIRST COUNT
            # TIMESTAMP and COUNTs arrays are equal
            times = []
            for ts in argo_data["PTSCI_TIMESTAMPS"][i, :nan_index]:
                times.append(datetime.datetime.strptime(str(ts)[:-2], "%Y%m%d%H%M%S"))
            times = np.array(times)
            time_deltas = []
            for dt in times:
                time_deltas.append(abs((dt - times[0]).total_seconds()))
            time_deltas = np.array(time_deltas)
            elptime = np.asarray(time_deltas)
        else:
            # KEEP FIRST COUNT
            # less trailing points, gets rid of last sum to keep shape
            elptime = np.cumsum(argo_data["NB_SAMPLE_CTD"][i, :nan_index])
            elptime = np.insert(elptime, 0, 0)[:-1]
        
        # Take data arrays
        temperature = argo_data["TEMP_ADJUSTED"][i, :nan_index]
        pressure = argo_data["PRES_ADJUSTED"][i, :nan_index]
        temp_cndc = argo_data["TEMP_CNDCs"][i, :nan_index]

        # Get rid of bad points 
        # WHERE COUNTS > 50: PSAL_ADJUSTED_QC, CNDC_ADJUSTED_QC, TEMP_ADJUSTED_QC are set to 3
        # 1) Get rid of bad points on all levels where COUNTS > 50
        # NOTE: if we have a level of ALL NaN's the RBR function gets weird... 
        count_mask = np.where(argo_data["NB_SAMPLE_CTD"][i, :nan_index] > 50)[0]
        temperature[count_mask] = np.nan
        #pressure[count_mask] = np.nan
        #temp_cndc[count_mask] = np.nan
        # 2) Get rid of bad points where TEMP, PRES, TEMP_CNDC QC arr's == (3/4)
        #temp_mask = np.where(((argo_data["TEMP_ADJUSTED_QC"][i, :nan_index] == 3) | (argo_data["TEMP_ADJUSTED_QC"][i, :nan_index] == 4)))[0]
        # pres_mask = np.where(((argo_data["PRES_ADJUSTED_QC"][i, :nan_index] == 3) | (argo_data["PRES_ADJUSTED_QC"][i, :nan_index] == 4)))[0]
        # temp_cndc_mask = np.where(((argo_data["TEMP_CNDC_QC"][i, :nan_index] == 3) | (argo_data["TEMP_CNDC_QC"][i, :nan_index] == 4)))[0]
        #temperature[temp_mask] = np.nan
        # pressure[pres_mask] = np.nan
        # temp_cndc[temp_cndc_mask] = np.nan

        a = RBRargo3_celltm.RBRargo3_celltm(temperature, pressure, temp_cndc, elptime)
        cell_tms.append(a)

    cell_tms = np.squeeze(np.array(list(itertools.zip_longest(*cell_tms, fillvalue=np.nan))).T)
    PSAL_ADJUSTED_Padj_CTM = gsw.SP_from_C(argo_data["CNDCs"] * 10, cell_tms, argo_data["PRES_ADJUSTED"])

    # apply qc_arr to data
    mask_var = np.isin(argo_data["PSAL_ADJUSTED_QC"], [0, 1, 2])
    mask_pres = np.isin(argo_data["PRES_ADJUSTED_QC"], [0, 1, 2])
    mask_temp = np.isin(argo_data["TEMP_ADJUSTED_QC"], [0, 1, 2])

    pres = np.where(mask_pres, argo_data["PRES_ADJUSTED"], np.nan)
    psal = np.where(mask_var, PSAL_ADJUSTED_Padj_CTM, np.nan)
    temp = np.where(mask_temp, argo_data["TEMP_ADJUSTED"], np.nan)
    
    #pres_v_var_all(pres, psal, argo_data["JULDs"], argo_data["PROFILE_NUMS"], "PSAL", float_num)
    #TS_graph_single_dataset_all_profile(psal, temp, argo_data["JULDs"], argo_data["LONs"], argo_data["LATs"], pres, argo_data["PROFILE_NUMS"], float_num)
    #single_prof_datasnapshot(285, argo_data, PSAL_ADJUSTED_Padj_CTM)
    make_der_graph(argo_data, PSAL_ADJUSTED_Padj_CTM)

def make_der_graph(argo_data, PSAL_ADJUSTED_Padj_CTM):

    for i in np.arange(len(argo_data["PROFILE_NUMS"])):
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(f'Profile: {argo_data["PROFILE_NUMS"][i]}')

        pres = argo_data["PRES_ADJUSTED"][i]
        psal_old = argo_data["PSAL_ADJUSTED"][i]
        psal_new = PSAL_ADJUSTED_Padj_CTM[i]

        # Plot salinity on left-side axis
        #ax1.scatter(psal_old, pres, s=1, c="red")
        ax1.plot(psal_old, pres, c="red")
        #ax1.scatter(psal_new, pres, s=1, c="green")
        ax1.plot(psal_new, pres, c="green")
        # Plot configs
        ax1.grid(True)
        ax1.yaxis.set_inverted(True)
        ax1.set_title("Adjusted v Original Salinity")
        ax1.set_xlabel("Salinity")
        ax1.set_ylabel("Pressure")
        ax1.legend(
            [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=7),
             Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=7)],
            ["Org Salinity", "Adjusted Salinity"], bbox_to_anchor=(0.5, -0.07), ncol = 2, loc='upper center'
        )

        # Plot derivative of temp/ depth on right side
        dT_dz = np.diff(argo_data["TEMP_ADJUSTED"][i]) / np.diff(pres) # divide the diff of adjacent elements of arrs
        pressure_mid = (pres[:-1] + pres[1:]) / 2 # Find the midpoint of pressures
        ax2.plot(dT_dz, pressure_mid)
        # Plot configs
        ax2.yaxis.set_inverted(True)
        ax2.set_title("dT/dz Graph")
        ax2.set_xlabel("dT/dz")
        ax2.set_ylabel("Pressure Midpoints")
        ax2.grid(True)
        ax2.axvline(0, color='black', linestyle='--', linewidth=1) # reference line

        plt.show()

def main():


    # make_thermal_inertia_graph()
    #make_thermal_inertia_graph_with_timestamps()
    compute_cellTM()

    raise Exception
   
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